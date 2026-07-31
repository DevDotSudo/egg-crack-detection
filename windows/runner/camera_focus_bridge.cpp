#include "camera_focus_bridge.h"

#include <windows.h>
#include <dshow.h>
#include <flutter/encodable_value.h>
#include <flutter/method_channel.h>
#include <flutter/standard_method_codec.h>

#include <algorithm>
#include <cmath>
#include <cwctype>
#include <memory>
#include <string>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#pragma comment(lib, "strmiids.lib")

namespace {

std::wstring Utf8ToWide(const std::string& value) {
  if (value.empty()) return L"";

  const int length = MultiByteToWideChar(
      CP_UTF8,
      0,
      value.c_str(),
      static_cast<int>(value.size()),
      nullptr,
      0);

  if (length <= 0) return L"";

  std::wstring result(length, L'\0');
  MultiByteToWideChar(
      CP_UTF8,
      0,
      value.c_str(),
      static_cast<int>(value.size()),
      result.data(),
      length);
  return result;
}

std::wstring ToLower(std::wstring value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](wchar_t character) {
        return static_cast<wchar_t>(std::towlower(character));
      });
  return value;
}

std::wstring ReadProperty(IPropertyBag* property_bag, const wchar_t* key) {
  if (property_bag == nullptr) return L"";

  VARIANT value;
  VariantInit(&value);
  std::wstring result;

  if (SUCCEEDED(property_bag->Read(key, &value, nullptr)) &&
      value.vt == VT_BSTR &&
      value.bstrVal != nullptr) {
    result = value.bstrVal;
  }

  VariantClear(&value);
  return result;
}

bool NameMatches(
    const std::wstring& requested,
    const std::wstring& friendly_name,
    const std::wstring& device_path) {
  const std::wstring requested_lower = ToLower(requested);
  const std::wstring friendly_lower = ToLower(friendly_name);
  const std::wstring path_lower = ToLower(device_path);

  if (requested_lower.empty()) {
    return friendly_lower.find(L"c525") != std::wstring::npos;
  }

  return (!friendly_lower.empty() &&
          (friendly_lower.find(requested_lower) != std::wstring::npos ||
           requested_lower.find(friendly_lower) != std::wstring::npos)) ||
      (!path_lower.empty() &&
       (path_lower.find(requested_lower) != std::wstring::npos ||
        requested_lower.find(path_lower) != std::wstring::npos)) ||
      (requested_lower.find(L"c525") != std::wstring::npos &&
       friendly_lower.find(L"c525") != std::wstring::npos);
}

enum class FocusAction {
  kStartAutoFocus,
  kLockCurrentFocus,
  kSetManualFocus,
};

bool ConfigureFocusOnFilter(
    IBaseFilter* camera_filter,
    FocusAction action,
    double normalized_position) {
  if (camera_filter == nullptr) return false;

  IAMCameraControl* camera_control = nullptr;
  if (FAILED(camera_filter->QueryInterface(IID_PPV_ARGS(&camera_control)))) {
    return false;
  }

  long minimum = 0;
  long maximum = 0;
  long step = 0;
  long default_value = 0;
  long supported_flags = 0;

  const HRESULT range_result = camera_control->GetRange(
      CameraControl_Focus,
      &minimum,
      &maximum,
      &step,
      &default_value,
      &supported_flags);

  if (FAILED(range_result)) {
    camera_control->Release();
    return false;
  }

  const bool supports_auto =
      (supported_flags & CameraControl_Flags_Auto) != 0;
  const bool supports_manual =
      (supported_flags & CameraControl_Flags_Manual) != 0;

  if ((action == FocusAction::kStartAutoFocus && !supports_auto) ||
      (action != FocusAction::kStartAutoFocus && !supports_manual)) {
    camera_control->Release();
    return false;
  }

  long current_value = default_value;
  long current_flags = 0;
  if (FAILED(camera_control->Get(
          CameraControl_Focus,
          &current_value,
          &current_flags))) {
    current_value = default_value;
  }

  HRESULT set_result = E_FAIL;
  if (action == FocusAction::kStartAutoFocus) {
    // Leaving the lens in manual mode briefly forces webcams such as the
    // Logitech C525 to begin a fresh autofocus sweep when auto is restored.
    if (supports_manual) {
      camera_control->Set(
          CameraControl_Focus,
          current_value,
          CameraControl_Flags_Manual);
      Sleep(40);
    }

    set_result = camera_control->Set(
        CameraControl_Focus,
        current_value,
        CameraControl_Flags_Auto);
  } else if (action == FocusAction::kLockCurrentFocus) {
    // Read the position reached by autofocus and hold it. A fixed inspection
    // distance should not run continuous AF because that causes focus hunting
    // between the egg and the background.
    set_result = camera_control->Set(
        CameraControl_Focus,
        current_value,
        CameraControl_Flags_Manual);
  } else {
    const double clamped_position = std::clamp(normalized_position, 0.0, 1.0);
    const long safe_step = std::max(step, 1L);
    const double requested_value =
        static_cast<double>(minimum) +
        (static_cast<double>(maximum - minimum) * clamped_position);
    long manual_value = minimum + static_cast<long>(std::llround(
        (requested_value - static_cast<double>(minimum)) /
        static_cast<double>(safe_step))) * safe_step;
    manual_value = std::clamp(manual_value, minimum, maximum);

    set_result = camera_control->Set(
        CameraControl_Focus,
        manual_value,
        CameraControl_Flags_Manual);
  }

  camera_control->Release();
  return SUCCEEDED(set_result);
}

bool TryConfigureFocus(
    IMoniker* camera_moniker,
    const std::wstring& requested_camera,
    bool require_match,
    FocusAction action,
    double normalized_position) {
  if (camera_moniker == nullptr) return false;

  IPropertyBag* property_bag = nullptr;
  std::wstring friendly_name;
  std::wstring device_path;

  if (SUCCEEDED(camera_moniker->BindToStorage(
          nullptr,
          nullptr,
          IID_PPV_ARGS(&property_bag)))) {
    friendly_name = ReadProperty(property_bag, L"FriendlyName");
    device_path = ReadProperty(property_bag, L"DevicePath");
    property_bag->Release();
  }

  if (require_match &&
      !NameMatches(requested_camera, friendly_name, device_path)) {
    return false;
  }

  if (!require_match &&
      ToLower(friendly_name).find(L"c525") == std::wstring::npos) {
    return false;
  }

  IBaseFilter* camera_filter = nullptr;
  if (FAILED(camera_moniker->BindToObject(
          nullptr,
          nullptr,
          IID_PPV_ARGS(&camera_filter)))) {
    return false;
  }

  const bool enabled = ConfigureFocusOnFilter(
      camera_filter,
      action,
      normalized_position);
  camera_filter->Release();
  return enabled;
}

bool EnumerateAndConfigure(
    IEnumMoniker* camera_enumerator,
    const std::wstring& requested_camera,
    bool require_match,
    FocusAction action,
    double normalized_position) {
  if (camera_enumerator == nullptr) return false;

  camera_enumerator->Reset();
  IMoniker* camera_moniker = nullptr;

  while (camera_enumerator->Next(1, &camera_moniker, nullptr) == S_OK) {
    const bool enabled = TryConfigureFocus(
        camera_moniker,
        requested_camera,
        require_match,
        action,
        normalized_position);
    camera_moniker->Release();

    if (enabled) return true;
  }

  return false;
}

bool ConfigureCameraFocus(
    const std::wstring& requested_camera,
    FocusAction action,
    double normalized_position = 0.5) {
  const HRESULT initialize_result =
      CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  const bool should_uninitialize = SUCCEEDED(initialize_result);

  if (FAILED(initialize_result) && initialize_result != RPC_E_CHANGED_MODE) {
    return false;
  }

  ICreateDevEnum* device_enumerator = nullptr;
  IEnumMoniker* camera_enumerator = nullptr;

  HRESULT result = CoCreateInstance(
      CLSID_SystemDeviceEnum,
      nullptr,
      CLSCTX_INPROC_SERVER,
      IID_PPV_ARGS(&device_enumerator));

  if (FAILED(result)) {
    if (should_uninitialize) CoUninitialize();
    return false;
  }

  result = device_enumerator->CreateClassEnumerator(
      CLSID_VideoInputDeviceCategory,
      &camera_enumerator,
      0);

  if (result != S_OK || camera_enumerator == nullptr) {
    device_enumerator->Release();
    if (should_uninitialize) CoUninitialize();
    return false;
  }

  bool enabled = EnumerateAndConfigure(
      camera_enumerator,
      requested_camera,
      true,
      action,
      normalized_position);

  if (!enabled) {
    enabled = EnumerateAndConfigure(
        camera_enumerator,
        requested_camera,
        false,
        action,
        normalized_position);
  }

  camera_enumerator->Release();
  device_enumerator->Release();

  if (should_uninitialize) CoUninitialize();
  return enabled;
}

}  // namespace

void RegisterCameraFocusBridge(flutter::BinaryMessenger* messenger) {
  static auto channel =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          messenger,
          "egg_camera_focus",
          &flutter::StandardMethodCodec::GetInstance());

  channel->SetMethodCallHandler(
      [](const auto& call, auto result) {
        const bool start_auto_focus =
            call.method_name() == "startAutoFocus" ||
            call.method_name() == "enableAutoFocus";
        const bool lock_current_focus =
            call.method_name() == "lockCurrentFocus";
        const bool set_manual_focus =
            call.method_name() == "setManualFocusPosition";

        if (!start_auto_focus && !lock_current_focus && !set_manual_focus) {
          result->NotImplemented();
          return;
        }

        std::string camera_name;
        double normalized_position = 0.5;
        const auto* arguments = std::get_if<flutter::EncodableMap>(
            call.arguments());

        if (arguments != nullptr) {
          const auto iterator = arguments->find(
              flutter::EncodableValue("cameraName"));

          if (iterator != arguments->end()) {
            const auto* value = std::get_if<std::string>(
                &iterator->second);
            if (value != nullptr) camera_name = *value;
          }

          const auto position_iterator = arguments->find(
              flutter::EncodableValue("position"));
          if (position_iterator != arguments->end()) {
            if (const auto* value = std::get_if<double>(
                    &position_iterator->second)) {
              normalized_position = *value;
            }
          }
        }

        const FocusAction action = start_auto_focus
            ? FocusAction::kStartAutoFocus
            : lock_current_focus
                ? FocusAction::kLockCurrentFocus
                : FocusAction::kSetManualFocus;

        result->Success(flutter::EncodableValue(
            ConfigureCameraFocus(
                Utf8ToWide(camera_name),
                action,
                normalized_position)));
      });
}

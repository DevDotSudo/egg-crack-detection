import 'package:file_selector/file_selector.dart';

/// Thin wrapper around file_selector so screens don't depend on the
/// package directly — makes it easy to swap the picker implementation
/// later without touching feature code.
class FileService {
  Future<XFile?> pickImage() async {
    const typeGroup = XTypeGroup(
      label: 'images',
      extensions: ['jpg', 'jpeg', 'png', 'bmp'],
    );
    return openFile(acceptedTypeGroups: [typeGroup]);
  }

  Future<List<XFile>> pickImages() async {
    const typeGroup = XTypeGroup(
      label: 'images',
      extensions: ['jpg', 'jpeg', 'png', 'bmp'],
    );
    return openFiles(acceptedTypeGroups: [typeGroup]);
  }

  Future<String?> pickDirectory() async {
    return getDirectoryPath();
  }

  Future<XFile?> saveFile({
    required String suggestedName,
    List<XTypeGroup> acceptedTypeGroups = const [],
  }) async {
    final location = await getSaveLocation(
      suggestedName: suggestedName,
      acceptedTypeGroups: acceptedTypeGroups,
    );
    return location == null ? null : XFile(location.path);
  }
}

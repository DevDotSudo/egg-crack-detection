import 'dart:async';
import 'dart:io';

import 'package:path/path.dart' as p;

class BackendLauncherService {
  Process? _process;

  List<String> _backendDirectories() {
    final cwd = Directory.current.path;
    final exeDir = p.dirname(Platform.resolvedExecutable);
    final candidates = <String>[
      p.join(cwd, 'backend'),
      p.join(exeDir, 'backend'),
      p.normalize(p.join(cwd, '..', 'backend')),
      p.normalize(p.join(exeDir, '..', 'backend')),
    ];

    var current = Directory(cwd);
    for (var i = 0; i < 7; i++) {
      candidates.add(p.join(current.path, 'backend'));
      final parent = current.parent;
      if (parent.path == current.path) break;
      current = parent;
    }

    return candidates.toSet().where((path) => Directory(path).existsSync()).toList();
  }

  String? _findBackendDirectory() {
    for (final directory in _backendDirectories()) {
      if (File(p.join(directory, 'run_server.py')).existsSync() ||
          File(p.join(directory, 'backend.exe')).existsSync()) {
        return directory;
      }
    }
    return null;
  }

  /// Find a virtual-environment Python given the backend directory.
  ///
  /// The venv may live:
  ///   1. Inside  backendDir/.venv/          (classic placement)
  ///   2. One level up  backendDir/../.venv/ (project-root placement)
  ///   3. Two levels up                      (monorepo / extra nesting)
  String? _findVenvPython(String backendDirectory) {
    final candidates = <String>[
      p.join(backendDirectory, '.venv', 'Scripts', 'python.exe'),
      p.normalize(p.join(backendDirectory, '..', '.venv', 'Scripts', 'python.exe')),
      p.normalize(p.join(backendDirectory, '..', '..', '.venv', 'Scripts', 'python.exe')),
    ];
    for (final candidate in candidates) {
      if (File(candidate).existsSync()) return candidate;
    }
    return null;
  }

  Future<bool> start() async {
    if (_process != null) return true;
    if (!Platform.isWindows) return false;

    final backendDirectory = _findBackendDirectory();
    if (backendDirectory == null) return false;

    final executable = p.join(backendDirectory, 'backend.exe');
    if (File(executable).existsSync() &&
        await _launch(executable, const [], backendDirectory)) {
      return true;
    }

    final script = p.join(backendDirectory, 'run_server.py');
    if (!File(script).existsSync()) return false;

    // Try the venv Python first (wherever it lives relative to backend dir).
    final venvPython = _findVenvPython(backendDirectory);
    if (venvPython != null &&
        await _launch(venvPython, [script], backendDirectory)) {
      return true;
    }

    // Fall back to system Python launchers.
    final launchers = <(String, List<String>)>[
      ('py', ['-3.11', script]),
      ('py', ['-3.13', script]),
      ('py', ['-3', script]),
      ('python', [script]),
    ];

    for (final launcher in launchers) {
      if (await _launch(launcher.$1, launcher.$2, backendDirectory)) {
        return true;
      }
    }

    return false;
  }

  Future<bool> _launch(
    String command,
    List<String> arguments,
    String workingDirectory,
  ) async {
    IOSink? logSink;
    try {
      final logFile = File(p.join(workingDirectory, 'backend_startup.log'));
      final sink = logFile.openWrite(mode: FileMode.writeOnlyAppend);
      logSink = sink;
      sink.writeln('\n--- Backend start ${DateTime.now().toIso8601String()} ---');
      sink.writeln('Command: $command ${arguments.join(' ')}');

      final process = await Process.start(
        command,
        arguments,
        workingDirectory: workingDirectory,
        runInShell: false,
        mode: ProcessStartMode.normal,
      );
      _process = process;

      process.stdout.transform(SystemEncoding().decoder).listen(sink.write);
      process.stderr.transform(SystemEncoding().decoder).listen(sink.write);

      final earlyExit = await Future.any<int?>([
        process.exitCode.then<int?>((code) => code),
        Future<int?>.delayed(const Duration(milliseconds: 700), () => null),
      ]);

      if (earlyExit != null) {
        sink.writeln('\nBackend exited with code $earlyExit');
        await sink.flush();
        await sink.close();
        if (_process == process) _process = null;
        return false;
      }

      process.exitCode.then((code) async {
        try {
          sink.writeln('\nBackend exited with code $code');
          await sink.flush();
          await sink.close();
        } catch (_) {}
        if (_process == process) _process = null;
      });
      return true;
    } catch (error) {
      try {
        await logSink?.close();
        final logFile = File(p.join(workingDirectory, 'backend_startup.log'));
        logFile.writeAsStringSync(
          '\nLaunch error: $error\n',
          mode: FileMode.append,
        );
      } catch (_) {}
      _process = null;
      return false;
    }
  }

  bool get isRunning => _process != null;
}

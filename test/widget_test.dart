import 'package:egg_crack_detection/app.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('shows the dashboard on startup', (tester) async {
    await tester.pumpWidget(const EggCrackDetectionApp());
    await tester.pump();

    expect(find.text('INSPECTION CONSOLE'), findsOneWidget);
    expect(find.text('SYSTEM INFO'), findsOneWidget);
  });
}

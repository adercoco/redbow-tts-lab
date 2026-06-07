import XCTest

final class RedBowVoiceUITests: XCTestCase {
    override func setUp() {
        continueAfterFailure = false
    }

    func testPressRecordAfterPermissionsDoesNotCrash() {
        let app = XCUIApplication()
        app.launch()

        let bow = app.descendants(matching: .any)["redBowRecordButton"]
        XCTAssertTrue(bow.waitForExistence(timeout: 5))

        addUIInterruptionMonitor(withDescription: "System permissions") { alert in
            for title in ["允許", "好", "OK", "Allow"] {
                let button = alert.buttons[title]
                if button.exists {
                    button.tap()
                    return true
                }
            }
            return false
        }

        bow.press(forDuration: 0.3)
        app.tap()
        XCTAssertEqual(app.state, .runningForeground)

        for _ in 0..<3 {
            bow.press(forDuration: 0.6)
            XCTAssertEqual(app.state, .runningForeground)
        }
    }
}

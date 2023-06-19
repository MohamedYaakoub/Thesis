import pywinauto
import pyautogui
import time

while True:
    try:
        # Wait for the dialog box to appear
        app = pywinauto.Application(backend="uia").connect(title_re="Error")  # Connect to the dialog window

        # Dismiss the dialog box by clicking 'OK' or an appropriate button
        dialog = app.window(title_re=".*")
        ok_button = dialog.child_window(title='OK', control_type='Button')
        ok_button.click_input()

        # Dismiss any subsequent confirmation dialog (if applicable)
        pyautogui.press('enter')

    except Exception as e:
        # Handle any other exceptions that may occur
        print("An error occurred:", str(e))

    # Sleep for a short duration before the next attempt
    time.sleep(1)
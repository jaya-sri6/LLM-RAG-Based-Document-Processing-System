from playwright.sync_api import sync_playwright, expect

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            # 1. Navigate to the app
            page.goto("http://localhost:5173")

            # 2. Initial state: "Get Analysis" button should be disabled
            get_analysis_button = page.get_by_role("button", name="Get Analysis")
            expect(get_analysis_button).to_be_disabled()
            page.screenshot(path="jules-scratch/verification/verification_step1_initial.png")

            # 3. Select a file
            file_input = page.locator('input[type="file"]')
            file_path = "jules-scratch/verification/sample.pdf"
            file_input.set_input_files(file_path)

            # 4. After file selection: "Get Analysis" button should still be disabled
            expect(get_analysis_button).to_be_disabled()
            page.screenshot(path="jules-scratch/verification/verification_step2_file_selected.png")

            # 5. Process the document
            process_button = page.get_by_role("button", name="Process")
            process_button.click()

            # 6. Wait for processing to complete and button to be enabled
            # The status text changes upon completion, which is a good indicator.
            expect(page.get_by_text("processed and ready for queries")).to_be_visible(timeout=30000) # 30s timeout
            expect(get_analysis_button).to_be_enabled()
            page.screenshot(path="jules-scratch/verification/verification_step3_processed.png")

            print("Verification script completed successfully.")

        except Exception as e:
            print(f"An error occurred during verification: {e}")
            page.screenshot(path="jules-scratch/verification/verification_error.png")
        finally:
            browser.close()

if __name__ == "__main__":
    run_verification()

from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("http://localhost:5173")
    page.locator('input[type="file"]').set_input_files('../data/sample_policy.txt')
    page.locator('button:has-text("Process")').click()
    expect(page.locator('button:has-text("Get Analysis")')).not_to_be_disabled(timeout=60000)
    page.fill('textarea', 'What is the policy number?')
    page.click('button:has-text("Get Analysis")')
    page.wait_for_selector('.bg-green-100')
    page.screenshot(path="../jules-scratch/verification/verification.png")
    browser.close()

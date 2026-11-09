import { expect, panes, test } from "./fixtures";

test("Editor view has the correct initial content", async ({ page }) => {
  const tab = panes(page).left.getByRole("tab", { name: "Script Editor", exact: true });
  await tab.click();
  await expect(tab).toHaveClass(/active/);

  const editor = page.locator("#code-wrapper");
  await expect(editor).toBeVisible();

  await expect(editor.locator(".view-line").first()).toHaveText('print("Welcome to the Bumps Editor!")');
});

test("Editor view can be reset to default content", async ({ page }) => {
  const tab = panes(page).left.getByRole("tab", { name: "Script Editor", exact: true });
  await tab.click();
  await expect(tab).toHaveClass(/active/);

  const editor = page.locator("#code-wrapper");
  await expect(editor).toBeVisible();

  // Change the content of the editor
  const line = editor.locator(".view-line").first();
  await line.click();
  await page.keyboard.type('print("Hello, World!")');
  await expect(line).not.toHaveText('print("Welcome to the Bumps Editor!")');

  // Click the reset button
  const resetButton = page.getByRole("button", { name: "Reset" });
  await resetButton.click();

  // Verify that the content has been reset to default
  await expect(line).toHaveText('print("Welcome to the Bumps Editor!")');
});

test("Editor view can be downloaded and uploaded", async ({ page }) => {
  const tab = panes(page).left.getByRole("tab", { name: "Script Editor", exact: true });
  await tab.click();
  await expect(tab).toHaveClass(/active/);

  const editor = page.locator("#code-wrapper");
  await expect(editor).toBeVisible();

  // Change the content of the editor
  const line = editor.locator(".view-line").first();
  await line.click();
  await page.keyboard.press("Control+A");
  await page.keyboard.type('print("Hello, World!")');
  await expect(line).toHaveText('print("Hello, World!")');

  // Click the download button
  const downloadButton = page.getByRole("button", { name: "Download" });
  const [download] = await Promise.all([page.waitForEvent("download"), downloadButton.click()]);

  // Save the downloaded file to a temporary path
  const downloadPath = await download.path();
  expect(downloadPath).not.toBeNull();

  // Click the reset button to clear the editor
  const resetButton = page.getByRole("button", { name: "Reset" });
  await resetButton.click();

  // Click the upload button and set the file to upload
  const uploadButton = page.getByRole("button", { name: "Upload" });
  const [chooser] = await Promise.all([page.waitForEvent("filechooser"), uploadButton.click()]);
  await chooser.setFiles(downloadPath);

  // Verify that the content has been restored from the uploaded file
  await expect(line).toHaveText('print("Hello, World!")');
});

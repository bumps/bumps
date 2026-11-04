import { expect, panes, test } from "./fixtures";

// The server is started with examples/curvefit/curve.py: a straight line with parameters m and b.

test("Loads the model and shows its parameters", async ({ page }) => {
  const { right } = panes(page);
  await right.getByRole("tab", { name: "Summary", exact: true }).click();
  const rows = right.locator("tbody tr");
  // listed alphabetically
  await expect(rows.locator("td:first-child")).toHaveText(["b", "m"]);
});

test("Data panel draws the model plot", async ({ page }) => {
  const { left } = panes(page);
  await left.getByRole("tab", { name: "Data", exact: true }).click();
  await expect(left.locator("#model-select option")).toHaveCount(1);
  await expect(left.locator(".tab-content svg").first()).toBeVisible();
});

for (const title of ["Data", "Summary", "Log", "History", "Convergence", "Model", "Parameters"]) {
  test(`${title} panel opens without errors`, async ({ page }) => {
    const { left } = panes(page);
    const tab = left.getByRole("tab", { name: title, exact: true });
    await tab.click();
    await expect(tab).toHaveClass(/active/);
    await expect(left.locator(".tab-content > *").first()).toBeVisible();
    // give async fetch-and-draw a moment to fail, if it's going to
    await page.waitForTimeout(500);
  });
}

test("Running a fit completes and is logged", async ({ page }) => {
  const { left } = panes(page);
  await left.getByRole("tab", { name: "Log", exact: true }).click();
  // wait for the log history to load before counting, or earlier fits arrive after `before` is read
  await expect(left.getByText(/^Model loaded:/)).toBeVisible();
  const done = left.locator("summary", { hasText: /^Done with chisq/ });
  const before = await done.count();

  await page.getByRole("button", { name: "Start" }).click();

  await expect(done).toHaveCount(before + 1, { timeout: 20_000 });
  // fit badge goes back to idle
  await expect(page.getByRole("button", { name: "Start" })).toBeVisible();
});

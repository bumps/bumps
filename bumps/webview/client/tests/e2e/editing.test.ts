import { expect, panes, test } from "./fixtures";

// The server is started with examples/curvefit/curve.py: a straight line with parameters m and b.

test("Editing a parameter updates chisq, and loading a saved history item restores it", async ({ page }) => {
  const { left, right } = panes(page);
  page.on("dialog", (dialog) => dialog.accept()); // "Reloading overwrites current state: continue?"

  await right.getByRole("tab", { name: "Summary", exact: true }).click();
  const mValue = right
    .locator("tbody tr", { has: page.getByRole("cell", { name: "m", exact: true }) })
    .locator("td")
    .nth(2);
  await expect(mValue).not.toBeEmpty();
  const originalValue = (await mValue.textContent())!.trim();

  await left.getByRole("tab", { name: "History", exact: true }).click();
  const chisq = left.locator("tr", { hasText: "Current state" }).locator("td").nth(2);
  await expect(chisq).not.toBeEmpty();
  const originalChisq = (await chisq.textContent())!.trim();

  // save the current state; the newest history item is listed first
  const items = left.locator("tbody tr", { has: page.locator(".history-name") });
  const before = await items.count();
  await left.getByRole("button", { name: "Save", exact: true }).click();
  await expect(items).toHaveCount(before + 1);
  const saved = items.first();
  const savedName = (await saved.locator(".history-name").textContent())!.trim();

  // edit m: the server recomputes chisq and sends the new value back
  await mValue.click(); // focusing selects the cell's contents
  await page.keyboard.type("5");
  await page.keyboard.press("Enter");
  await expect(chisq).not.toHaveText(originalChisq);
  await expect(mValue).not.toHaveText(originalValue);

  // load the saved item: parameters and chisq go back to how they were
  const savedItem = items.filter({ has: page.getByText(savedName, { exact: true }) });
  await savedItem.getByRole("button", { name: "Load" }).click();
  await expect(savedItem.getByText("Loaded")).toBeVisible();
  await expect(mValue).toHaveText(originalValue);
  await expect(chisq).toHaveText(originalChisq);

  // clean up so other tests see the history as it was
  await savedItem.getByRole("button", { name: "Close" }).click();
  await expect(items).toHaveCount(before);
});

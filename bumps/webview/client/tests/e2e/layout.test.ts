import type { Page } from "@playwright/test";
import { expect, panes, test } from "./fixtures";

async function setLayout(page: Page, layout: "left-right" | "top-bottom" | "full") {
  await page.getByRole("button", { name: "Layout", exact: true }).click();
  await page.getByRole("button", { name: layout, exact: true }).click();
}

/** Active tab of each pane, in order, regardless of layout. */
const activeTabs = (page: Page) => page.locator(".nav-tabs [role=tab].active");

test("Selected tabs survive layout changes", async ({ page }) => {
  const { left, right } = panes(page);
  await left.getByRole("tab", { name: "Log", exact: true }).click();
  await right.getByRole("tab", { name: "History", exact: true }).click();
  await expect(activeTabs(page)).toHaveText(["Log", "History"]);

  await setLayout(page, "top-bottom");
  await expect(activeTabs(page)).toHaveText(["Log", "History"]);

  await setLayout(page, "full");
  await expect(activeTabs(page)).toHaveText(["Log"]);

  // the second pane is recreated here, and should come back on the tab it had
  await setLayout(page, "left-right");
  await expect(activeTabs(page)).toHaveText(["Log", "History"]);
});

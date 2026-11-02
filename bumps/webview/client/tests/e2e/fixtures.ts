import { test as base, expect } from "@playwright/test";

/**
 * `page` that has loaded the app and finished connecting to the server, and
 * fails the test if the page throws or logs a console error.
 */
export const test = base.extend<{ errors: string[] }>({
  errors: [
    async ({ page }, use) => {
      const errors: string[] = [];
      page.on("pageerror", (err) => errors.push(err.message));
      page.on("console", (msg) => {
        if (msg.type() === "error") errors.push(msg.text());
      });
      await use(errors);
      expect(errors, "page errors / console errors").toEqual([]);
    },
    { auto: true },
  ],
  page: async ({ page }, use) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Server Connecting..." })).toBeHidden();
    await use(page);
  },
});

/** Left and right panel areas in the default left-right layout. */
export const panes = (page: import("@playwright/test").Page) => {
  const columns = page.locator(".col").filter({ has: page.locator(".nav-tabs") });
  return { left: columns.nth(0), right: columns.nth(1) };
};

export { expect };

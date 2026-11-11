import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// jsdom has no modal <dialog>; the source drawer only needs these to exist.
if (!HTMLDialogElement.prototype.showModal) {
  HTMLDialogElement.prototype.showModal = function () {
    this.open = true;
  };
  HTMLDialogElement.prototype.close = function () {
    this.open = false;
  };
}

afterEach(cleanup);

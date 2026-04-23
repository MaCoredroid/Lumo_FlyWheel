
import helper from "./helper.cjs";

export const plugin = {
  name: "good-helper",
  run(input) {
    return `${helper.renderLabel("good-helper")}:${input}`;
  }
};

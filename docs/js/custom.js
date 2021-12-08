/**
 * Declare config
 */
const config = {
  api: "https://phodexr.war.ocf.io",
};

/**
 * jQuery to attach navbar for semantic
 */

$(document).ready(function () {
  // fix menu when passed
  $(".masthead").visibility({
    once: false,
    onBottomPassed: function () {
      $(".fixed.menu").transition("fade in");
    },
    onBottomPassedReverse: function () {
      $(".fixed.menu").transition("fade out");
    },
  });

  // create sidebar and attach to menu open
  $(".ui.sidebar").sidebar("attach events", ".toc.item");
});

const createParagraph = (text) => {
  const paragraphNode = document.createElement("p");
  paragraphNode.textContent = text;
  return paragraphNode;
};

window.onload = (_event) => {
  /**
   * Demo Form
   */
  const HOCNNDemoForm = document.getElementById("hocnn-demo");

  const submitHOCNNDemoForm = async (e) => {
    e.preventDefault();

    const formData = new FormData(HOCNNDemoForm);

    const predictions = await fetch(config.api + "hocnn/predict", {
      method: "POST",
      body: formData,
    });
    const parsedPredictions = await predictions.json();
    const HOCNNResultsNode = document.getElementById("hocnn-results");
    HOCNNResultsNode.innerHTML = "";

    const parsedPredictionsKeys = Object.keys(parsedPredictions);
    for (let i = 0; i < parsedPredictionsKeys.length; i++) {
      const predictionKey = parsedPredictionsKeys[i];
      if (predictionKey == "fasterrcnn_object") {
        const predictionText =
          "Faster-RCNN object: " + parsedPredictions[predictionKey];
        HOCNNResultsNode.appendChild(createParagraph(predictionText));
      } else {
        const predictionText =
          "" +
          (i + 1) +
          " " +
          predictionKey +
          " " +
          parsedPredictions[predictionKey] +
          "%";
        HOCNNResultsNode.appendChild(createParagraph(predictionText));
      }
    }

    return false;
  };

  HOCNNDemoForm.addEventListener("submit", submitHOCNNDemoForm, false);
};

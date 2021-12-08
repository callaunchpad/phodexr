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
  const CLIPDemoForm = document.getElementById("clip-demo");

  const submitCLIPDemoForm = async (e) => {
    e.preventDefault();

    const formData = new FormData(CLIPDemoForm);

    const imageEmbedding = await fetch(config.api + "/api/vision", {
      method: "POST",
      body: formData,
    });
    const parsedImageEmbedding = await imageEmbedding.json();

    const scores = [];

    const dot = (arr1, arr2) => {
      let total = 0;

      for (let i = 0; i < arr1.length; i++) {
        total += arr1[i] * arr2[i];
      }

      return total;
    };

    for (let i = 1; i <= 3; i++) {
      const textElem = document.getElementById("text" + i);
      const textEmbedding = await fetch(
        config.api + "/api/nlp?text=" + encodeURIComponent(textElem.value)
      );
      const parsedTextEmbedding = await textEmbedding.json();

      scores.push({
        score: dot(
          parsedImageEmbedding.embedding,
          parsedTextEmbedding.embedding
        ),
        caption: textElem.value,
      });
    }

    const sortedScores = scores.sort((a, b) => b.score - a.score);

    const CLIPresultsNode = document.getElementById("clip-results");
    CLIPresultsNode.innerHTML = "";

    const captionPrediction = scores[0].caption;
    const predictionText = "Most likely caption: " + captionPrediction;

    CLIPresultsNode.appendChild(createParagraph(predictionText));

    return false;
  };

  CLIPDemoForm.addEventListener("submit", submitCLIPDemoForm, false);
};

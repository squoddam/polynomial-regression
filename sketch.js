// --> VARS
const x_vals = [];
const y_vals = [];

const SIDE = 600;

const NUM_OF_OPTIONS = 4;
const MIN_DEGREE = 2;
const OPTIONS_HEIGHT = 50;

const VARIABLES_WIDTH = 300;
const VARIABLES_MARGIN = 30;
const A_CHAR_CODE = "a".charCodeAt();

let degree = 2;
let selectedDegree = degree;

let variablesStore = new Array(MIN_DEGREE + NUM_OF_OPTIONS - 1)
  .fill(null)
  .map(() => tf.variable(tf.scalar(0)));

let variables = [];

const curveX = [];

for (let x = -1; x <= 1; x += 0.01) {
  curveX.push(x);
}

const learningRate = 0.8;
const optimizer = tf.train.adam(learningRate);
// <-- VARS

// --> UTILS
const updateVariables = () => {
  const newVariables = [];
  for (let i = 0; i < degree; i += 1) {
    // this way we won't have memory leak, creating new tensors
    if (variables[i]) {
      newVariables.push(variables[i]);

      // synchronize
      variablesStore[i] = variables[i];
    } else {
      newVariables.push(variablesStore[i]);
    }
  }

  variables = newVariables;
};

const customMap = (val, { isHeight, isReversed } = {}) => {
  const H = height - OPTIONS_HEIGHT;
  const bounds = isHeight ? [1, -1] : [-1, 1];
  const dimension = isHeight ? H : width;
  const leftValue = isHeight ? 0 : VARIABLES_WIDTH;

  if (isReversed) {
    return map(val, ...bounds, leftValue, dimension);
  }

  return map(val, leftValue, dimension, ...bounds);
};

// <-- UTILS

// --> FOR PREDICTIONS

function predict(x_vals) {
  return tf.tidy(() => {
    const xs = tf.tensor1d(x_vals);

    const ys = variables.reduce((res, variable, i) => {
      const power = variables.length - 1 - i;

      return res.add(res.pow(power).mul(variable));
    }, xs);

    return ys;
  });
}
// <-- FOR PREDICTIONS

// --> P5
function setup() {
  createCanvas(SIDE + VARIABLES_WIDTH, SIDE + OPTIONS_HEIGHT);
  updateVariables();
}

function mouseClicked() {
  // points placement
  const x = customMap(mouseX);
  const y = customMap(mouseY, { isHeight: true });

  if (x >= -1 && x <= 1 && y >= -1 && y <= 1) {
    x_vals.push(x);
    y_vals.push(y);
  }

  // degree selection
  if (
    mouseY > height - OPTIONS_HEIGHT &&
    mouseY < height &&
    mouseX > VARIABLES_WIDTH &&
    mouseX < width
  ) {
    const optionIndex = Math.floor(
      (mouseX - VARIABLES_WIDTH) / ((width - VARIABLES_WIDTH) / NUM_OF_OPTIONS)
    );

    selectedDegree = optionIndex + MIN_DEGREE;
  }
}

function draw() {
  // update degree if needed
  if (degree !== selectedDegree) {
    degree = selectedDegree;
    updateVariables();
  }

  // optimize variables
  tf.tidy(() => {
    if (x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals);
      optimizer.minimize(
        () => tf.losses.huberLoss(predict(x_vals), ys),
        false,
        variables
      );
    }
  });

  // start actual draw
  background(0);

  fill(255);

  stroke(255);
  strokeWeight(8);

  // POINTS
  x_vals.forEach((x, i) => {
    const px = customMap(x, { isReversed: true });
    const py = customMap(y_vals[i], { isHeight: true, isReversed: true });

    point(px, py);
  });

  // LINE
  const ys = predict(curveX);
  const curveY = ys.dataSync();
  ys.dispose();

  beginShape();
  noFill();
  strokeWeight(2);

  curveX.forEach((cx, i) => {
    const x = customMap(cx, { isReversed: true });
    const y = customMap(curveY[i], { isHeight: true, isReversed: true });

    vertex(x, y);
  });

  endShape();

  // --> DEGREE SELECTOR
  const widthForSelector = width - VARIABLES_WIDTH;
  const heightForSelector = height - OPTIONS_HEIGHT;

  fill(255);
  rect(VARIABLES_WIDTH, heightForSelector, widthForSelector, OPTIONS_HEIGHT);
  stroke(0);
  textSize(18);
  textAlign(CENTER, CENTER);

  for (let i = 0; i < NUM_OF_OPTIONS; i += 1) {
    const optionWidth = widthForSelector / NUM_OF_OPTIONS;
    const optionX = VARIABLES_WIDTH + i * optionWidth;
    const optionY = height - OPTIONS_HEIGHT + 1;

    if (degree === i + MIN_DEGREE) {
      rect(optionX, optionY, optionWidth, OPTIONS_HEIGHT - 4);
      fill(0);
    }

    text(
      i + MIN_DEGREE,
      optionX + optionWidth / 2,
      optionY + OPTIONS_HEIGHT / 2
    );

    fill(255);
  }
  // <-- DEGREE SELECTOR

  // --> FUNC AND VARIABLES INFO
  strokeWeight(0);
  rect(0, 0, VARIABLES_WIDTH, height);

  fill(0);
  let fBase = [];

  for (let i = 0; i < degree; i += 1) {
    const letter = String.fromCharCode(A_CHAR_CODE + i);
    const power = degree - 1 - i;

    if (power === 1) {
      fBase.push(`${letter}x`);
    } else if (power === 0) {
      fBase.push(letter);
    } else {
      fBase.push(`${letter}x^${power}`);
    }
  }

  text(`y = ${fBase.join(" + ")}`, VARIABLES_WIDTH / 2, VARIABLES_MARGIN);

  textAlign(LEFT, TOP);
  variables.forEach((variable, i) => {
    text(
      `${String.fromCharCode(A_CHAR_CODE + i)}: ${variable.dataSync()[0]}`,
      VARIABLES_MARGIN,
      VARIABLES_MARGIN * (i + 2)
    );
  });
  // <-- FUNC AND VARIABLES INFO
}
// <-- P5

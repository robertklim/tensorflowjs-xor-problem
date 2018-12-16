let model;

// visualization resolution
let resolution = 50;
let cols;
let rows;

// model inputs
let xs;

// data for XOR
// [0, 1] -> [1]
// [1, 0] -> [1]
// [0, 0] -> [0]
// [1, 1] -> [0]
const train_xs = tf.tensor2d([
    [0, 1],
    [1, 0],
    [0, 0],
    [1, 1]
]);

const train_ys = tf.tensor2d([
    [1],
    [1],
    [0],
    [0]
]);

function setup() {
    createCanvas(400, 400);

    // visualization variables
    cols = width / resolution;
    rows = height / resolution;

    // create the input data
    let inputs = [];
    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            let x1 = i / cols;
            let x2 = j / rows;
            inputs.push([x1, x2]);
        }
    }
    // create model inputs
    xs = tf.tensor2d(inputs);

    // define tf model:
    // 2 inputs
    // 2 hidden nodes
    // 1 output
    
    model = tf.sequential();
    
    let hidden = tf.layers.dense({
        inputShape: [2],
        units: 2,
        activation: 'sigmoid'
    });
    
    let output = tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    });
    
    model.add(hidden);
    model.add(output);

    const optimizer = tf.train.sgd(0.5);
    
    model.compile({
        optimizer: optimizer,
        loss: 'meanSquaredError'
    });

}

async function trainModel() {
    return await model.fit(train_xs, train_ys, {
        shuffle: true
    });
}

function draw() {
    background(0);

    // training
    tf.tidy(() => {
        trainModel().then(result => {
            console.log(result.history.loss[0]);
        });
    });
    // noLoop();

    // get predictions
    tf.tidy(() => {
        let ys = model.predict(xs);
        let y_values = ys.dataSync();
    
        // draw
        let index = 0;
        for (let i = 0; i < cols; i++) {
            for (let j = 0; j < rows; j++) {
                let bright = y_values[index] * 255;                
                fill(bright);
                noStroke();
                rect(i * resolution, j * resolution, resolution, resolution);
                fill(255 - bright);
                noStroke();
                textAlign(CENTER, CENTER);
                text(nf(y_values[index], 1, 2), i * resolution + resolution / 2, j * resolution + resolution / 2);
                index++;
            }
        }
    });
    noLoop();

    // labels
    l = '[0, 0]';
    textSize(16);
    fill(255, 0, 0);
    //stroke(255);
    text(l, 32, 16);

    l = '[1, 0]';
    textSize(16);
    fill(255, 0, 0);
    //stroke(255);
    text(l, width - 36, 16);

    l = '[1, 1]';
    textSize(16);
    fill(255, 0, 0);
    //stroke(255);
    text(l, width - 36, height - 10);

    l = '[0, 1]';
    textSize(16);
    fill(255, 0, 0);
    //stroke(255);
    text(l, 32, height - 10);

}
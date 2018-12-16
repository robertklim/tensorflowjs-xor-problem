let model;

// data for XOR
// [0, 1] -> [1]
// [1, 0] -> [1]
// [0, 0] -> [0]
// [1, 1] -> [0]
let training_data = [
    {
        inputs: [0, 1],
        targets: [1]
    },
    {
        inputs: [1, 0],
        targets: [1]
    },
    {
        inputs: [0, 0],
        targets: [0]
    },
    {
        inputs: [1, 1],
        targets: [0]
    },
];

function setup() {
    createCanvas(400, 400);

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

    const optimizer = tf.train.sgd(0.1);
    
    model.compile({
        optimizer: optimizer,
        loss: 'meanSquaredError'
    });

}

function draw() {
    background(0);

    // visualization
    let resolution = 10;
    let cols = width / resolution;
    let rows = height / resolution;
    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            let x1 = i / cols;
            let x2 = j / rows;
            let y = x1 * x2;
            fill(y * 255);
            noStroke();
            rect(i * resolution, j * resolution, resolution, resolution);
        }
    }

    // labels
    l = '[0, 0]';
    textSize(16);
    fill(255, 0, 0);
    stroke(255);
    text(l, 0, 16);

    l = '[1, 0]';
    textSize(16);
    fill(255, 0, 0);
    stroke(255);
    text(l, width - 36, 16);

    l = '[1, 1]';
    textSize(16);
    fill(255, 0, 0);
    stroke(255);
    text(l, width - 36, height - 8);

    l = '[0, 1]';
    textSize(16);
    fill(255, 0, 0);
    stroke(255);
    text(l, 0, height - 8);

}
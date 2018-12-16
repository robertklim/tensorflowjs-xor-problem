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

}

function draw() {
    background(0);
}
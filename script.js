// create data
var data_shirt = [
    {
        "height" : "1561",
        "size" : "S"
    },
    {
        "height" : "1673",
        "size" : "M"
    },
    {
        "height" : "1794",
        "size" : "L"
    },
    {
        "height" : "1801",
        "size" : "XL"
    },
    {
        "height" : "1802",
        "size" : "XL"
    },
    {
        "height" : "1803",
        "size" : "XL"
    }
]
// normal data
var height = []
var size_name = ["S", "M", "L", "XL"]
var sizes = []

data_shirt.map((item) => {
    height.push((item.height - 1400) / 500);
    sizes.push(item.size)
})

let xs = tf.tensor1d(height);
let sizeTensor = tf.tensor1d(sizes, 'int32')
let ys = tf.oneHot(sizeTensor, 5)
// create model
let model = tf.sequential();
let hidden = tf.layers.dense({
    units: 16,
    activation: "sigmoid",
    inputDim: 1
})

let output = tf.layers.dense({
    units: 5,
    activation: "softmax",
})

model.add(hidden)
model.add(output)

model.compile({
    optimizer: tf.train.sgd(0.2),
    loss: 'categoricalCrossentropy'
})



// show data frontend

async function train() {
    let options = {
        epochs: 10,
        validationSplit: 0.1,
        shuffle: true
    }
    
    // train model
    return await model.fit(xs, ys, options)
}

console.log('Tranning start')

train().then((result) => {
    console.log('tranning finish')
    let heightTextBox = document.getElementById('height');
    let getBtnResult = document.getElementById('getResult');
    let resultText = document.getElementById('resultText');

    getBtnResult.addEventListener('click', function() {
        let height = heightTextBox.value;
        let heightInput = (height - 1400) / 500;
        let heightInputTensor = tf.tensor1d([heightInput])

        let predictResult = model.predict(heightInputTensor);

        let max = predictResult.argMax(1).dataSync()[0];

        resultText.innerHTML = 'Size ao cua ban la ' + size_name[max]
    })
})
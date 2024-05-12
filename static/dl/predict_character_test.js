const canvas = document.getElementById('main-canvas');
const smallCanvas = document.getElementById('small-canvas');
const smallMainCanvas = document.getElementById('small-main-canvas');
const displayBox = document.getElementById('prediction');

const inputBox = canvas.getContext('2d');
const smBox = smallCanvas.getContext('2d');
const smMBox = smallMainCanvas.getContext('2d');

let isDrawing = false;
let model;

async function init() {  
  model = await tf.loadModel('/static/dl/tfjs-models/model-character/model.json');
}


canvas.addEventListener('mousedown', event => {
  isDrawing = true;

  inputBox.strokeStyle = 'white';
  inputBox.lineWidth = '15';
  inputBox.lineJoin = inputBox.lineCap = 'round';
  inputBox.beginPath();
});

canvas.addEventListener('mousemove', event => {
  if (isDrawing) drawStroke(event.clientX, event.clientY);
});

canvas.addEventListener('mouseup', event => {
  isDrawing = false;
});

function drawStroke(clientX, clientY) {
  // get mouse coordinates on canvas
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;

  // draw
  inputBox.lineTo(x, y);
  inputBox.stroke();
  inputBox.moveTo(x, y);
}

/* Makes predictions */
function predict() {
  let values = getPixelData(flipAndRotate());
  let predictions = model.predict(values).dataSync();

  return predictions;
}

function getPixelData(imgData) {

  let values = [];
  for (let i = 0; i < imgData.data.length; i += 4) {
    values.push(imgData.data[i] / 255);
  }
  values = tf.reshape(values, [1, 64, 64, 1]);
  return values;
}

function give_verdict() {
  digit = document.getElementById("writingDigit").textContent;
  smMBox.drawImage(inputBox.canvas, 0, 0, smallCanvas.width, smallCanvas.height);
  const imageData = smMBox.getImageData(0, 0, smallCanvas.width, smallCanvas.height);
  
  // Find index of best prediction, which corresponds to the predicted value
  predictions = predict()
  // console.log(predictions);
  const bestPred = Math.max(...predictions);
  const bestPredIndex = predictions.indexOf(Math.max(...predictions));

  if(digit == bestPredIndex && Math.floor(bestPred*10)>=3) {
    displayBox.innerText = `أَحْسَنْت يَا بَطَل! تَحَصَّلْتَ عَلَى ${Math.floor(bestPred*10)} مِنْ 10`;
    console.log("Bravo");
  }
  else {
    displayBox.innerText = `خاطئ! مَا رَأْيُكَ فِي مُحَاوَلَة أُخْرَى يَا بَطَل؟`;
    console.log("Not bravo");
  }
}

document.getElementById('erase').addEventListener('click', erase);
document.getElementById('predict').addEventListener('click', give_verdict);

function flipAndRotate() {
    smBox.save();
    smBox.translate(smallCanvas.width/2,smallCanvas.height/2);
    smBox.rotate(270*Math.PI/180);
    smBox.scale(-1, 1);
    smBox.drawImage(smMBox.canvas,-smallCanvas.width/2,-smallCanvas.height/2);
    smBox.restore();
    return smBox.getImageData(0, 0, smallCanvas.width, smallCanvas.height);
}

function erase() {
  inputBox.fillStyle = 'black';
  inputBox.fillRect(0, 0, canvas.width, canvas.height);
  displayBox.innerText = '';
  label.innerText = '';
}

function drawRotated(){
    smBox.clearRect(0,0,smallCanvas.width,smallCanvas.height);
    smBox.save();
    smBox.translate(smallCanvas.width/2,smallCanvas.height/2);
    smBox.rotate(-90*Math.PI/180);
    smBox.drawImage(image,-smallCanvas.width/2,-smallCanvas.height/2);
    smBox.restore();
}

	var start = function(e) {
    isDrawing = true;


    inputBox.strokeStyle = 'white';
    inputBox.lineWidth = '15';
    inputBox.lineJoin = inputBox.lineCap = 'round';
		inputBox.beginPath();
	};
	var move = function(e) {
		e.preventDefault();
		x = e.changedTouches[0].pageX;
		y = e.changedTouches[0].pageY-44;
    if (isDrawing) drawStroke(e.changedTouches[0].pageX, e.changedTouches[0].pageY);
	};

  var end = function(e) {
    isDrawing = false;
  };
  document.getElementById("main-canvas").addEventListener("touchstart", start, false);
  document.getElementById("main-canvas").addEventListener("touchmove", move, false);
  document.getElementById("main-canvas").addEventListener("touchend", end, false);

  function getRandomInt(max) {
    return Math.floor(Math.random() * max);
  }

  function random_digit() {
    var digit = getRandomInt(28);
    document.getElementById("writingDigit").textContent = digit;
    document.getElementById("writingDigit_text_arb").textContent = ARB_CHAR_PRONOUNCE_CLASS[digit];
    document.getElementById('audio_source').src=`/static/dl/audio/character/${digit}.mp3`;
    document.getElementById('audio').load();
    document.getElementById('audio').play();
  }


erase();
init();
random_digit();
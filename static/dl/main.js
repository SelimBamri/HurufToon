const ARB_NUMBER_CLASSES = {
    0: 'صِفْــر',
    1: 'وَاحِــدْ',
    2: 'اثْـنَـانْ',
    3: 'ثـلَاثَـة',
    4: 'أَرْبَــعَــة',
    5: 'خَـمْــسَةٌ',
    6: 'سِِـــتَّــةٌ',
    7: 'سَــبْــعَـةٌ',
    8: 'ثَـمَانِـيَةٌ',
    9: 'تِـسْـعَـة'
}
var canvas,
  ctx,
  canvasColor = "black";
var lineWidth = 15,
  strokeColor = "yellow";
var mouseX,
  mouseY,
  isMouseDown = false;
var touchX, touchY;
var isResultDivPresent = false;
let model;
//the base url of website in which our web app is deployed is obtained from window.location.origin the json file is loaded using async function
var base_url = window.location.origin;
var to_guesss_number = 0;
/* Loading the Model */

(async function () {
  console.log("Model loading...");
  // for local testing
  model = await tf.loadLayersModel(
    "https://swagatobag2000.github.io/digit-recognition-webapp/models/model.json"
  );
  console.log("Model loaded..");
})();

/* fucnction for interacting with canvas */

function init() {
  canvas = document.getElementById("sketchpad");
  ctx = canvas.getContext("2d"); // '2d' means two dimensional rendering context on canvas
  ctx.fillStyle = canvasColor; // now we will fill ctx background with black
  ctx.fillRect(0, 0, canvas.width, canvas.height); // fill the rect
  console.log("Canvas Width: ", canvas.width, canvas.height);
  to_guesss_number = Math.floor(Math.random() * 10);
  legend = document.getElementById("to_guess");
  legend.textContent = ARB_NUMBER_CLASSES[to_guesss_number];
  if (ctx) {
    //if mousedown than call function sketchPad_mouseDown.false means bubble phase
    canvas.addEventListener("mousedown", sketchpad_mouseDown, false);
    canvas.addEventListener("mousemove", sketchpad_mouseMove, false);
    canvas.addEventListener("mouseup", sketchpad_mouseUp, false);
    canvas.addEventListener("touchstart", sketchpad_touchStart, false);
    canvas.addEventListener("touchmove", sketchpad_touchMove, false);
  }
}

/* now to enable drawing on canvas we define draw function */

function drawWithMouse(ctx, x, y, size, isDown) {
  if (isDown) {
    ctx.beginPath(); //to inform canvas, user is about to draw
    ctx.strokeStyle = strokeColor; //to set background color of canvas
    ctx.lineWidth = lineWidth; //set width of line

    // .linejoin() : set connection between two line,
    // .lineCap() : to set end of line
    ctx.lineJoin = ctx.lineCap = "round";
    ctx.moveTo(lastX, lastY); //it tells where to start drawing line
    ctx.lineTo(x, y); //draw line from start to current position of pointer
    ctx.closePath(); //drawing is complete
    ctx.stroke(); //to paint the line drawn with some pixel
  }
  // else mousedown than start posi is curr position
  lastX = x;
  lastY = y;
}

/*Event handlers*/

/* when mouse is down it will call draw function */
function sketchpad_mouseDown(event) {
  isMouseDown = true;
  drawWithMouse(ctx, mouseX, mouseY, 12, false);
}

/* when mouse is released it set's mousedown back to false and start prediction */

const result = document.getElementById("predict");
result.addEventListener('click', async () => {
  isMouseDown = false;
  var imageData = canvas.toDataURL();
  let tensor = preprocessCanvas(canvas);
  console.log("Tensor after preprocessing: ", tensor);
  let predictions = await model.predict(tensor).data();
  let results = Array.from(predictions);
  displayLabel(results);
})
async function sketchpad_mouseUp(event) {
  isMouseDown = false;
}

/* when mouse is moved in either direction it gets current position of mouse from getMousePos(e) and if mouseDown than call draw */

function sketchpad_mouseMove(event) {
  getMousePos(event);
  if (isMouseDown) {
    drawWithMouse(ctx, mouseX, mouseY, 12, true);
  }
}

/* finds current position of pointer
when mouse event is triggered, offset x,offset y -> return x,y cordinate of mouse
and layer x,layer y -> return horizantak and vertical cordinates relative to current layer */

function getMousePos(event) {
  if (event.offsetX) {
    mouseX = event.offsetX;
    mouseY = event.offsetY;
  } else if (event.layerX) {
    mouseX = event.layerX;
    mouseY = event.layerY;
  }
}

/* touch event handler */

/* when user touches the touchpad it calls draw func with false to note position not to draw */
function sketchpad_touchStart(event) {
  getTouchPos(event);
  drawWithMouse(ctx, touchX, touchY, 12, false);
  event.preventDefault(); //this prevents scrolling of screen when user draws
}

/* when user drags in sketchpad it calls draw with true flag to enable drawing */

function sketchpad_touchMove(event) {
  getTouchPos(event);
  drawWithMouse(ctx, touchX, touchY, 12, true);
  event.preventDefault();
}

/* it is used to find point in the sketchpad where user has touched */

function getTouchPos(event) {
  if (event.touches) {
    if (event.touches.length == 1) {
      //its length is used to find  how many fingers has touched
      var touch = event.touches[0];
      touchX = touch.pageX - touch.target.offsetLeft;
      touchY = touch.pageY - touch.target.offsetTop;
    }
  }
}

/* clearing the sketchpad */
function clear() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = canvasColor;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  clearAll();
  isResultDivPresent = false;
}

document.getElementById("clear_button").addEventListener("click", clear);

/*preprocessing image from canvas*/

/* resizing the input image to target size of (1, 28, 28) 
tf.browser.fromPixels() method -> to create a tensor that will flow into the first layer of the model
tf.image.resizeNearestNeighbor() -> resizes a batch of 3D images to a new shape
tf.mean() -> used to compute the mean of elements across the dimensions of the tensor
tf.toFloat() -> casts the array to type float
tensor.div() -> used to divide the array or tensor by the maximum RGB value(255)
*/

function preprocessCanvas(image) {
  let tensor = tf.browser
    .fromPixels(image)
    .resizeNearestNeighbor([28, 28])
    .mean(2)
    .expandDims(2)
    .expandDims()
    .toFloat();
  return tensor.div(255.0);
}

/*Clear the previous Output*/

function clearAll() {
  const result_div_list = document.getElementById("result");
  while (result_div_list.firstChild) {
    result_div_list.removeChild(result_div_list.lastChild);
  }
  isResultDivPresent = false;
  var tdList = document.getElementsByTagName("td");
  for (var i = 0; i < tdList.length; i++) {
    tdList[i].innerHTML = "";
    if (tdList[i].classList.contains("answer")) {
      tdList[i].classList.remove("answer");
      document.getElementsByTagName("th")[i + 2].classList.remove("answer");
    }
  }
}

/* output and labelling*/

async function displayLabel(data) {
  var maxElement = Math.max(...data);
  var maxIndex = data.indexOf(maxElement);

  console.log("Max Element: ", maxElement);
  console.log("Max Element Index: ", maxIndex);
  if(maxIndex === to_guesss_number){
    texte = `أَحْسَنْت يَا بَطَل! تَحَصَّلْتَ عَلَى ${Math.floor(maxElement*10)} مِنْ 10`;
    console.log(texte)
  }
  else {
    texte = `خاطئ! مَا رَأْيُكَ فِي مُحَاوَلَة أُخْرَى يَا بَطَل؟`;
  }
  label_disp = document.getElementById('label');
  label_disp.textContent = texte;
}

const audio_button_ar = document.getElementById("aud_ar")
const synth = window.speechSynthesis;
    let utterance = new SpeechSynthesisUtterance();

    audio_button_ar.addEventListener('click', () =>{
      utterance.lang = 'ar-SA';
      utterance.text = ARB_NUMBER_CLASSES[to_guesss_number];
      synth.speak(utterance);
    })

const audio_button_fr = document.getElementById("aud_fr")
    audio_button_fr.addEventListener('click', () =>{
      utterance.lang = 'fr-FR';
      utterance.text = to_guesss_number;
      synth.speak(utterance);
    })

const audio_button_en = document.getElementById("aud_eng")
    audio_button_en.addEventListener('click', () =>{
      utterance.lang = 'en-US';
      utterance.text = to_guesss_number;
      synth.speak(utterance);
    })

const next_button = document.getElementById("tryAnother");
    next_button.addEventListener('click', () =>{
      to_guesss_number = Math.floor(Math.random() * 10);
      legend = document.getElementById("to_guess");
      legend.textContent = ARB_NUMBER_CLASSES[to_guesss_number];
      label_disp = document.getElementById('label');
      label_disp.textContent = "";
      clear();
    })
let draggableObjects;
let dropPoints;
const startButton = document.getElementById("start");
const result = document.getElementById("result");
const controls = document.querySelector(".controls-container");
const dragContainer = document.querySelector(".draggable-objects");
const dropContainer = document.querySelector(".drop-points");
const data = [
    "bang",
    "friend",
    "goodluck",
    "help",
    "hi",
    "no",
    "ok",
    "pay",
    "love",
    "play",
    "please",
    "rock",
    "shocker",
    "thanks",
    "week",
    "yes",
    "baby",
    "signlanguage",
    "eat",
    "sorry",
];

let deviceType = "";
let initialX = 0,
  initialY = 0;
let currentElement = "";
let moveElement = false;

// Detect touch device
const isTouchDevice = () => {
  try {
    document.createEvent("TouchEvent");
    deviceType = "touch";
    return true;
  } catch (e) {
    deviceType = "mouse";
    return false;
  }
};

let count = 0;

// Random value from Array
const randomValueGenerator = () => {
  return data[Math.floor(Math.random() * data.length)];
};

// Win Game Display
const stopGame = () => {
  controls.classList.remove("hide");
  startButton.classList.remove("hide");

  // Add an event listener to the redirect button
  const redirectButton = document.getElementById("home-btn");
  redirectButton.style.display = "block";
  redirectButton.addEventListener("click", function() {
      console.log("Redirecting to lesson.html...");
      window.location.href = "lesson.html";
  });
};

// Drag & Drop Functions
function dragStart(e) {
  if (isTouchDevice()) {
    initialX = e.touches[0].clientX;
    initialY = e.touches[0].clientY;
    moveElement = true;
    currentElement = e.target;
  } else {
    e.dataTransfer.setData("text", e.target.id);
  }
}

// Events fired on the drop target
function dragOver(e) {
  e.preventDefault();
}

// For touchscreen movement
const touchMove = (e) => {
  if (moveElement) {
    e.preventDefault();
    let newX = e.touches[0].clientX;
    let newY = e.touches[0].clientY;
    let currentSelectedElement = document.getElementById(e.target.id);
    currentSelectedElement.parentElement.style.top =
      currentSelectedElement.parentElement.offsetTop - (initialY - newY) + "px";
    currentSelectedElement.parentElement.style.left =
      currentSelectedElement.parentElement.offsetLeft - (initialX - newX) + "px";
    initialX = newX;
    initialY = newY;
  }
};

const drop = (e) => {
  e.preventDefault();
  if (isTouchDevice()) {
    moveElement = false;
    const currentDrop = document.querySelector(`div[data-id='${e.target.id}']`);
    const currentDropBound = currentDrop.getBoundingClientRect();
    if (
      initialX >= currentDropBound.left &&
      initialX <= currentDropBound.right &&
      initialY >= currentDropBound.top &&
      initialY <= currentDropBound.bottom
    ) {
      currentDrop.classList.add("dropped");
      currentElement.classList.add("hide");
      currentDrop.innerHTML = ``;
      currentDrop.insertAdjacentHTML(
        "afterbegin",
        `<img src="static/img/${currentElement.id}.png">`
      );
      count += 1;
    }
  } else {
    const draggedElementData = e.dataTransfer.getData("text");
    const droppableElementData = e.target.getAttribute("data-id");
    if (draggedElementData === droppableElementData) {
      const draggedElement = document.getElementById(draggedElementData);
      e.target.classList.add("dropped");
      draggedElement.classList.add("hide");
      draggedElement.setAttribute("draggable", "false");
      e.target.innerHTML = ``;
      e.target.insertAdjacentHTML(
        "afterbegin",
        `<img src="static/img/${draggedElementData}.png">`
      );
      count += 1;
    }
  }
  if (count == 3) {
    result.innerText = `You Won!`;
    stopGame();
  }
};

const creator = () => {
  dragContainer.innerHTML = "";
  dropContainer.innerHTML = "";
  let randomData = [];
  for (let i = 1; i <= 3; i++) {
    let randomValue = randomValueGenerator();
    if (!randomData.includes(randomValue)) {
      randomData.push(randomValue);
    } else {
      i -= 1;
    }
  }
  for (let i of randomData) {
    const flagDiv = document.createElement("div");
    flagDiv.classList.add("draggable-image");
    flagDiv.setAttribute("draggable", true);
    if (isTouchDevice()) {
      flagDiv.style.position = "absolute";
    }
    flagDiv.innerHTML = `<img src="static/img/${i}.png" id="${i}">`;
    dragContainer.appendChild(flagDiv);
  }
  randomData = randomData.sort(() => 0.5 - Math.random());
  for (let i of randomData) {
    const countryDiv = document.createElement("div");
    countryDiv.innerHTML = `<div class='countries' data-id='${i}'>
      ${i.charAt(0).toUpperCase() + i.slice(1).replace("-", " ")}
    </div>`;
    dropContainer.appendChild(countryDiv);
  }
};

// Event listener for the "click" event on the startButton
startButton.addEventListener("click", async () => {
  currentElement = "";
  controls.classList.add("hide");
  startButton.classList.add("hide");
  await creator();
  count = 0;
  dropPoints = document.querySelectorAll(".countries"); // Assign class "countries" to the dropPoints variable
  draggableObjects = document.querySelectorAll(".draggable-image"); // Assign class "draggable-image" to the draggableObjects variable

  // Attach event listeners to each draggable image
  draggableObjects.forEach((element) => {
    element.addEventListener("dragstart", dragStart);
    element.addEventListener("touchstart", dragStart);
    element.addEventListener("touchend", drop);
    element.addEventListener("touchmove", touchMove);
  });

  // Attach event listeners to each drop point
  dropPoints.forEach((element) => {
    element.addEventListener("dragover", dragOver);
    element.addEventListener("drop", drop);
  });
});

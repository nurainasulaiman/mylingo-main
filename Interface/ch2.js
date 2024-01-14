let timer = document.getElementsByClassName("timer")[0];
let quizContainer = document.getElementById("container");
let nextButton = document.getElementById("next-button");
let numOfQuestions = document.getElementsByClassName("number-of-questions")[0];
let displayContainer = document.getElementById("display-container");
let scoreContainer = document.querySelector(".score-container");
let restart = document.getElementById("restart");
let userScore = document.getElementById("user-score");
let startScreen = document.querySelector(".start-screen");
let startButton = document.getElementById("start-button");
let questionCount;
let scoreCount = 0;
let count = 10;
let countdown;

const Data = [
  "img/bang.png",
  "img/friend.png",
  "img/goodluck.png",
  "img/help.png",
  "img/hi.png",
  "img/no.png",
  "img/ok.png",
  "img/pay.png",
  "img/love.png",
  "img/play.png",
  "img/please.png",
  "img/rock.png",
  "img/shocker.png",
  "img/thanks.png",
  "img/week.png",
  "img/yes.png",
  "img/baby.png",
  "img/signlanguage.png",
  "img/eat.png",
  "img/sorry.png",
];

//Questions and Options Array
let quizArray = [];

const generateRandomValue = (array) =>
  array[Math.floor(Math.random() * array.length)];

const populateOptions = (optionsArray, Data) => {
  let expectedLength = 4;
  while (optionsArray.length < expectedLength) {
    let randomIndex = Math.floor(Math.random() * Data.length);
    let imagePath = Data[randomIndex];
    if (!optionsArray.includes(imagePath)) {
      optionsArray.push(imagePath);
    }
  }
  return optionsArray;
};

const populateQuiz = () => {
  for (let i = 0; i < 5; i++) {
    let currentDataIndex = Math.floor(Math.random() * Data.length);
    let currentData = Data[currentDataIndex];
    let correctImage = Data[currentDataIndex];

    let allImages = [];
    allImages.push(correctImage);
    allImages = populateOptions(allImages, Data);

    quizArray.push({
      id: i,
      correct: correctImage,
      options: allImages,
    });
  }
};

nextButton.addEventListener(
  "click",
  (displayNext = () => {
    questionCount += 1; //increment questionCOunt
    if (questionCount == quizArray.length) {  //If last question, hide question container and display score
      displayContainer.classList.add("hide");
      scoreContainer.classList.remove("hide");
      userScore.innerHTML =   //User score
        "Your score is " + scoreCount + " out of " + questionCount;
    } else {
      numOfQuestions.innerHTML =
        questionCount + " of " + quizArray.length + " Question";

      quizDisplay(questionCount);
      count = 10;
      clearInterval(countdown);
      timerDisplay();
    }
    nextButton.classList.add("hide");
  })
);

//Timer
const timerDisplay = () => {
  countdown = setInterval(() => {
    timer.innerHTML = `<span>Time Left: </span> ${count}s`;
    count--;
    if (count == 0) {
      clearInterval(countdown);
      displayNext();
    }
  }, 1000);
};

const quizDisplay = (questionCount) => {
  let quizCards = document.querySelectorAll(".container-mid");
  if (quizCards.length > 0 && quizCards[questionCount]) {
    quizCards.forEach((card) => {
      card.classList.add("hide");
    });

    quizCards[questionCount].classList.remove("hide");
  }
};

function quizCreator() {
  quizArray.sort(() => Math.random() - 0.5);

  for (let i of quizArray) {
    i.options.sort(() => Math.random() - 0.5);

    let div = document.createElement("div");
    div.classList.add("container-mid", "hide");

    numOfQuestions.innerHTML = 1 + " of " + quizArray.length + " Question";

    let questionDiv = document.createElement("p");
    questionDiv.classList.add("question");
    let imageName = i.correct.split("/").pop().split(".")[0];
    questionDiv.textContent = imageName;
    div.appendChild(questionDiv);

    div.innerHTML += `
      <div class="button-container">
        <button class="option-div" onclick="checker(this)" style="background-image: url('${i.options[0]}'); background-size: 136px;" Data-option="${i.options[0]}"></button>
        <button class="option-div" onclick="checker(this)" style="background-image: url('${i.options[1]}'); background-size: 136px;" Data-option="${i.options[1]}"></button>
        <button class="option-div" onclick="checker(this)" style="background-image: url('${i.options[2]}'); background-size: 136px;" Data-option="${i.options[2]}"></button>
        <button class="option-div" onclick="checker(this)" style="background-image: url('${i.options[3]}'); background-size: 136px;" Data-option="${i.options[3]}"></button>
      </div>
    `;
    quizContainer.appendChild(div);
  }
}

function checker(userOption) {

let userSolution = userOption.getAttribute("Data-option");
let question = document.getElementsByClassName("container-mid")[questionCount];
let options = question.querySelectorAll(".option-div");

options.forEach((element) => {
  element.disabled = true; // Disable all buttons
});

if (userSolution === quizArray[questionCount].correct) {
  userOption.classList.add("correct");
  scoreCount++;
} else {
  userOption.classList.add("incorrect");
  options.forEach((element) => {
    if (element.getAttribute("Data-option") == quizArray[questionCount].correct) {
      element.classList.add("correct");
    } else {
      element.classList.remove("correct", "incorrect");
    }
  });
}

//clear interval
clearInterval(countdown);
//disable all options
options.forEach((element) => {
  element.disabled = true;
});
nextButton.classList.remove("hide");
}



function addClickHandlers() {
  let optionButtons = document.querySelectorAll('.option-div');
  optionButtons.forEach(button => {
    button.addEventListener('click', function () {
      checker(this);
    });
  });
}

function initial() {
  nextButton.classList.add("hide");
  quizContainer.innerHTML = "";
  questionCount = 0;
  scoreCount = 0;
  clearInterval(countdown);
  count = 10;
  timerDisplay();
  quizCreator();
  quizDisplay(questionCount);
  addClickHandlers(); // Add event handlers dynamically
}

restart.addEventListener("click", () => {
  quizArray = [];
  populateQuiz();
  initial();
  displayContainer.classList.remove("hide");
  scoreContainer.classList.add("hide");
});

startButton.addEventListener("click", () => {
  startScreen.classList.add("hide");
  displayContainer.classList.remove("hide");
  quizArray = [];
  populateQuiz();
  initial();
});

initial();

const moves = document.getElementById("moves-count");
const timeValue = document.getElementById("time");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const gameContainer = document.querySelector(".game-container");
const result = document.getElementById("result");
const controls = document.querySelector(".controls-container");
let cards;
let interval;
let firstCard = false;
let secondCard = false;

//Items array
const items = [
  { name: "bang", image: "static/img/bang.png" },
  { name: "friend", image: "static/img/friend.png" },
  { name: "goodluck", image: "static/img/goodluck.png" },
  { name: "help", image: "static/img/help.png" },
  { name: "no", image: "static/img/no.png" },
  { name: "ok", image: "static/img/ok.png" },
  { name: "pay", image: "static/img/pay.png" },
  { name: "play", image: "static/img/play.png" },
  { name: "rock", image: "static/img/rock.png" },
  { name: "shocker", image: "static/img/shocker.png" },
  { name: "week", image: "static/img/week.png" },
  { name: "yes", image: "static/img/yes.png" },
];

//Initial Time
let seconds = 0,
  minutes = 0;
//Initial moves and win count
let movesCount = 0,
  winCount = 0;

//For timer
function startCountdown() {
  const countdownElement = document.getElementById("time");

  // Set the total countdown time in seconds
  let totalSeconds =  60; // 5 minutes

  function updateCountdown() {
    // Calculate remaining minutes and seconds
    let minutes = Math.floor(totalSeconds / 60);
    let seconds = totalSeconds % 60;

    // Format the time before displaying
    let minutesValue = minutes < 10 ? `0${minutes}` : minutes;
    let secondsValue = seconds < 10 ? `0${seconds}` : seconds;

    // Display the countdown
    countdownElement.innerHTML = `<span>Time Left:</span> ${minutesValue}:${secondsValue}`;

    // Check if the countdown has reached zero
    if (totalSeconds === 0) {
      clearInterval(countdownInterval);
      console.log("Countdown is complete!");
      stopGame();
      result.innerHTML = `<h2>Times up!!</h2>`

    } else {
      // Decrement the total seconds
      totalSeconds--;
    }
  }

  // Call the updateCountdown function every second
  const countdownInterval = setInterval(updateCountdown, 1000);

}

//For calculating moves
const movesCounter = () => {
  movesCount += 1;
  moves.innerHTML = `<span>Moves:</span>${movesCount}`;
};

//Pick random objects from the items array
const generateRandom = (size = 4) => {
  //temporary array
  let tempArray = [...items];
  //initializes cardValues array
  let cardValues = [];
  //size should be double (4*4 matrix)/2 since pairs of objects would exist
  size = (size * size) / 2;
  //Random object selection
  for (let i = 0; i < size; i++) {
    const randomIndex = Math.floor(Math.random() * tempArray.length);
    cardValues.push(tempArray[randomIndex]);
    //once selected remove the object from temp array
    tempArray.splice(randomIndex, 1);
  }
  return cardValues;
};

const matrixGenerator = (cardValues, size = 4) => {
  gameContainer.innerHTML = "";
  cardValues = [...cardValues, ...cardValues];
  //simple shuffle
  cardValues.sort(() => Math.random() - 0.5);
  for (let i = 0; i < size * size; i++) {


    gameContainer.innerHTML += `
      <div class="card-container" data-card-value="${cardValues[i].name}">
          <div class="card-before">?</div>
          <div class="card-after">
              <img src="${cardValues[i].image}" class="image" style="width: 86px; height: 86px;"/>
          </div>
      </div>`;
  }

  //Grid
  gameContainer.style.gridTemplateColumns = `repeat(${size},auto)`;

  //Cards
  cards = document.querySelectorAll(".card-container");
  cards.forEach((card) => {
    card.addEventListener("click", () => {
      //If selected card is not matched yet then only run (i.e already matched card when clicked would be ignored)
      if (!card.classList.contains("matched")) {
        //flip the clicked card
        card.classList.add("flipped");
        //if it is the firstcard (!firstCard since firstCard is initially false)
        if (!firstCard) {
          //so current card will become firstCard
          firstCard = card;
          //current cards value becomes firstCardValue
          firstCardValue = card.getAttribute("data-card-value");
        } else {
          //increment moves since user selected second card
          movesCounter();
          //secondCard and value
          secondCard = card;
          let secondCardValue = card.getAttribute("data-card-value");
          if (firstCardValue == secondCardValue) {
            //if both cards match add matched class so these cards would beignored next time
            firstCard.classList.add("matched");
            secondCard.classList.add("matched");
            //set firstCard to false since next card would be first now
            firstCard = false;
            //winCount increment as user found a correct match
            winCount += 1;
            //check if winCount ==half of cardValues
            if (winCount == Math.floor(cardValues.length / 2)) {
              result.innerHTML = `<h2>You Won</h2>
            <h4>Moves: ${movesCount}</h4>`;
              stopGame();
            }
            } else {
              //if the cards dont match
              //flip the cards back to normal
              let [tempFirst, tempSecond] = [firstCard, secondCard];
              firstCard = false;
              secondCard = false;
              let delay = setTimeout(() => {
                tempFirst.classList.remove("flipped");
                tempSecond.classList.remove("flipped");
              }, 900);
            }
        }
      }
    });
  });
};

//Start game
startButton.addEventListener("click", () => {
  movesCount = 0;
  seconds = 0;
  minutes = 0;
  //controls amd buttons visibility
  controls.classList.add("hide");
  stopButton.classList.remove("hide");
  startButton.classList.add("hide");
  //Start timer
  interval = startCountdown();
  
  //initial moves
  moves.innerHTML = `<span>Moves:</span> ${movesCount}`;
  initializer();

});

//Stop game
stopButton.addEventListener(
  "click",
  (stopGame = () => {
    controls.classList.remove("hide");
    stopButton.classList.add("hide");
    startButton.classList.remove("hide");
    clearInterval(interval);

      // Add an event listener to the redirect button
        const redirectButton = document.getElementById("home-btn");
        redirectButton.style.display = "block";
        redirectButton.addEventListener("click", function() {
        console.log("Redirecting to lesson.html...");
        window.location.href = "lesson.html";
    });
  })
);

//Initialize values and func calls
const initializer = () => {
  result.innerText = "";
  winCount = 0;
  let cardValues = generateRandom();
  console.log(cardValues);
  matrixGenerator(cardValues);
};
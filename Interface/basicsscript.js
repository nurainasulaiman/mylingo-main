const questionsContainer = [
    {
        question: [
            { name: "", image: "img/hi.png" },
        ],
        answers: [
            {text: "Thank You", correct: false},
            {text: "Hi!", correct: true},
            {text: "Yes", correct: false},
            {text: "No", correct: false},
            {text: "Please", correct: false},
        ]
    },
    {
        question: [
            { name: "", image: "img/no.png" },
        ],
        answers: [
            {text: "Thank You", correct: false},
            {text: "Hi!", correct: false},
            {text: "Yes", correct: false},
            {text: "No", correct: true},
            {text: "Please", correct: false},
        ]
    },
    {
        question: [
            { name: "", image: "img/thanks.png" },
        ],
        answers: [
            {text: "Thank You", correct: true},
            {text: "Hi!", correct: false},
            {text: "Yes", correct: false},
            {text: "No", correct: false},
            {text: "Please", correct: false},
        ]
    },
    {
        question: [
            { name: "", image: "img/yes.png" },
        ],
        answers: [
            {text: "Thank You", correct: false},
            {text: "Hi!", correct: false},
            {text: "Yes", correct: true},
            {text: "No", correct: false},
            {text: "Please", correct: false},
        ]
    },
    {
        question: [
            { name: "", image: "img/please.png" },
        ],
        answers: [
            {text: "Thank You", correct: false},
            {text: "Hi!", correct: false},
            {text: "Yes", correct: false},
            {text: "No", correct: false},
            {text: "Please", correct: true},
        ]
    }
]

const questionElement = document.getElementById("question");
const questionContainer = document.getElementById("questionContainer");

const answerButtons = document.getElementById("answer-buttons");
const nextButton = document.getElementById("next-btn");
const btnElements = document.getElementsByClassName("btn");
const homeButton = document.getElementById("home-btn");

let currentQuestionIndex = 0;
let score = 0;

function startQuiz(){
    currentQuestionIndex = 0;
    score = 0;
    nextButton.innerHTML = "Next";
    displayQuestionAndImage(currentQuestionIndex);
}
startQuiz();

function displayQuestionAndImage() {
    const questionContainer = document.getElementById('questionContainer');

    // Get the current question object
    const currentQuestion = questionsContainer[currentQuestionIndex];
    console.log(currentQuestion);

    // Display the question text
    questionContainer.innerHTML = currentQuestion.question[0].name;

    // Create an img element for the question's image
    var imageElement = document.getElementById("qImage");

    // Set the size of the image to 96px
    imageElement.width = 96;
    imageElement.height = 96;
    
    console.log(imageElement);
    imageElement.src = currentQuestion.question[0].image;
    imageElement.alt = currentQuestion.question[0].name;
    imageElement.style.display = "block";
    imageElement.style.margin = "0 auto";

    // Append the image to the question div
    // questionContainer.appendChild(imageElement);

    let questionNo = currentQuestionIndex + 1;
    // questionElement.innerHTML = questionNo + ". " + currentQuestion.question[0].name;
    // questionElement.innerHTML = questionNo = ". " + currentQuestion.question[0].name;


    displayAnswers(currentQuestion.answers);
}

function displayAnswers(answers) {
    // Clear existing buttons
    answerButtons.innerHTML = "";

    // Loop through each answer and create a button
    answers.forEach((answer, index) => {
        const button = document.createElement("button");
        button.innerHTML = answer.text;
        button.classList.add("btn");
        answerButtons.appendChild(button);
        console.log(currentQuestionIndex);
        // Attach a click event listener to the button
        button.addEventListener("click", () => {
            selectAnswer(button, index); // Pass the index of the selected answer
            
        });
    });
}

function selectAnswer(button, selectedIndex) {
    const selectedAnswer = questionsContainer[currentQuestionIndex].answers[selectedIndex];
    const isCorrect = selectedAnswer.correct;

    // Add your logic for handling the selected answer (e.g., updating score)
    if (isCorrect) {
        console.log("Correct answer!");
        button.classList.add("correct");


        score++;
    } else {
        button.classList.add("incorrect");
        console.log("Incorrect answer!");
        const correctButtonIndex = questionsContainer[currentQuestionIndex].answers.findIndex(answer => answer.correct);
        const correctButton = answerButtons.children[correctButtonIndex];
        correctButton.classList.add("correct");
    }

    // Disable all buttons after an answer is selected
    disableButtons();

    // Show the "Next" button
    showNextButton();
}

function disableButtons() {
    // Disable all answer buttons
    Array.from(answerButtons.children).forEach(button => {
        button.disabled = true;
    });
}

function showNextButton() {
    nextButton.style.display = "block";
}

function showScore() {
    questionContainer.innerHTML = `You scored ${score} out of ${questionsContainer.length}!`;
    qImage.style.display = "none";

    // Hide existing buttons
    for (let i = 0; i < btnElements.length; i++) {
        btnElements[i].style.display = "none";
    }

    // Create "Play Again" button
    nextButton.innerHTML = "Play Again";
    nextButton.style.display = "block";

    // Add an event listener to the redirect button
    const redirectButton = document.getElementById("home-btn");
    redirectButton.style.display = "block";
    redirectButton.addEventListener("click", function() {
        console.log("Redirecting to parallax.html...");
        window.location.href = "parallax.html";
    });
}

function handleNextButton() {
    currentQuestionIndex++;
    if (currentQuestionIndex < questionsContainer.length) {
        displayQuestionAndImage();
    } else {
        showScore();
    }
}

nextButton.addEventListener("click", () => {
    if (currentQuestionIndex < questionsContainer.length) {
        handleNextButton();
    } else {
        startQuiz();
    }
});
const express = require('express');
const session = require('express-session');
const bodyParser = require('body-parser');

const app = express();

// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";  
import { getDatabase, set, ref, update } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-database.js";  // Updated import

// FIREBASE CONFIGURATION

const firebaseConfig = {
  apiKey: "AIzaSyCFnG26z5ZGcZuM3WUCG8CHznUg6x69D3k",
  authDomain: "signlingua-cbafd.firebaseapp.com",
  projectId: "signlingua-cbafd",
  storageBucket: "signlingua-cbafd.appspot.com",
  messagingSenderId: "970842735126",
  appId: "1:970842735126:web:a25a84d1a40b3928c96b37",
  measurementId: "G-1PZYENSFGY"
};
  
// Initialize Firebase
const firebaseApp = initializeApp(firebaseConfig);
const auth = getAuth(firebaseApp);
const database = getDatabase(firebaseApp);

app.use(bodyParser.urlencoded({ extended: true }));
app.use(session({
    secret: 'your-secret-key',
    resave: false,
    saveUninitialized: true
}));

app.get('/', (req, res) => {
    const user = req.session.user;
    res.send(`<h1>Hello, ${user ? user.username : 'Guest'}!</h1>`);
});

app.post('/register', (req, res) => {
    // Handle user registration and store information in session
    const { username, email } = req.body;
    req.session.user = { username, email };
    res.redirect('/');
});

app.post('/submitData', (req, res) => {
  var email = req.body.email;
  var password = req.body.psw;

  createUserWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      const user = userCredential.user;
      set(ref(database, 'users/' + user.uid), {
        email: email,
        password: password
      })
      .then(() => {
        const lgDate = new Date();
        update(ref(database, 'users/' + user.uid), {
          last_login: lgDate,
        })
        .then(() => {
          // Data saved successfully!
          res.send('User created and logged in successfully');
        })
        .catch((error) => {
          // The write failed...
          res.status(500).send(error.message);
        });
      })
      .catch((error) => {
        // The write failed...
        res.status(500).send(error.message);
      });
    })
    .catch((error) => {
      res.status(500).send(error.message);
    });
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});

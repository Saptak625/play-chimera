let gameContainer = document.getElementById("game-container");
let decisionDiv = document.getElementById("decisions");

let BET_OPTIONS = ["Pass", "1", "2", "3", "4", "Shoot"];
let SUIT_OPTIONS = ["Hearts", "Diamonds", "Clubs", "Spades"];
let SUIT_OPTIONS_2 = ["H", "D", "C", "S"];
let RANK_OPTIONS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"];
let players = [];
let gameManager;

function setup() {
	// Attach canvas to the specific DOM container
	let canvas = createCanvas(800, 800);
	canvas.parent(gameContainer);
	
	// Initialize 4 players: 0=Bottom, 1=Left, 2=Top, 3=Right
	// In Cinch, standard dealing gives 9 cards to each player
	players.push(new Player("Player 1 (You)", "bottom", HAND_SIZE));
	players.push(new Player("Player 2 (CHIMERA)", "left", HAND_SIZE));
	players.push(new Player("Player 3 (You)", "top", HAND_SIZE));
	players.push(new Player("Player 4 (CHIMERA)", "right", HAND_SIZE));

    gameManager = new GameManager(players);

}

async function draw() {
	background(34, 139, 34); // Classic card table green
	
    // Display the game state (players, cards, etc.)
	gameManager.display();
}

// Make a game manager to handle game state, turns, and interactions in the future
class GameManager {
    constructor(players) {
        this.waitingOnPlayer = false; // Whether we're waiting on the user to make a decision
        this.decisionOptions = []; // Store the possible decisions for the user to click on when it's their turn
        this.playerDecision = null; // Store the user's decision when they make one
        this.bettingPhase = true; // Whether we're in the betting phase or the playing phase
        this.readyForNextStep = true; // A flag to control the pacing of the game loop and ensure we wait for API responses before proceeding to the next step
        this.players = players;
        this.score = [0, 0]; // Team scores: [Team 1 (Players 1 & 3), Team 2 (Players 2 & 4)]
        this.round = 1;
        this.lastBettingGameState = null;
        this.bettingPlayer = null;
        this.setupComplete = false;
        this.message = "";

        // Randomly assign the dealer (for demonstration, we can just set it to player 0)
        this.dealer = random([0, 1, 2, 3]);
        this.currentPlayer = (this.dealer + 1) % 4; // The player to the left of the dealer starts first

        // Start the game by calling the API to initialize the game state.
        this.startGame();
    }

    async startGame() {
        try {
            let jsonData = await this.postToAPI('/api/start-game', { dealer: this.dealer }); 
            this.lastBettingGameState = jsonData;
            await this.startNewRound(); 
            this.setupComplete = true; 
            this.done = false;

            // Kick off the dedicated, sequential game loop!
            this.runGameLoop();
        } catch (error) {
            console.error("Failed to start game:", error);
        }
    }

    // This loop executes entirely independent of the p5.js draw framerate
    async runGameLoop() {
        while (this.setupComplete) {
            // Wait for the current game step/API call to fully complete
            await this.update();
            
            // Add a slight pacing pause (e.g., 100ms) to keep from hammering the CPU
            // while waiting on player choices or processing game turns
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    async startNewRound() {
        this.round = this.lastBettingGameState.rounds_played;
        this.score[0] = this.lastBettingGameState.points_team[0];
        this.score[1] = this.lastBettingGameState.points_team[1];
        for (let i = 0; i < this.players.length; i++) {
            this.players[i].setCards(this.lastBettingGameState.starting_hands[i]);
        }
        this.dealer = this.lastBettingGameState.dealer;
        this.players.forEach(player => player.setDealer(false)); // Reset all players to not be the dealer
        this.players[this.dealer].setDealer(true);
        this.currentPlayer = this.lastBettingGameState.current_player;
        this.bettingPlayer = this.lastBettingGameState.betting_player;
        this.bettingPhase = this.lastBettingGameState.main_game === null; // If main_game is null, we're still in the betting phase
        this.bet = null;
        this.trumpSuit = null;
        this.trumpsAtStart = [];
        this.countInWidow = null;
    }

    async bettingPhaseUpdate() {
        try {
            let copyOfGameState = JSON.parse(JSON.stringify(this.lastBettingGameState)); // Create a deep copy of the last betting game state
            let jsonData;

            if (this.bettingPlayer === null || this.lastBettingGameState.main_game !== null) {
                // We're still in the betting phase, so we need to send the player's bet decision to the API and update the game state based on the response.
                // Or we just finished the main game and need to compile the results into the score. In this case the playerDecision will be [null, null].
                // If player decision is a string (e.g. "Pass", "Shoot"), we need to convert it to the corresponding index that the API expects.
                if (typeof this.playerDecision === "string") {
                    copyOfGameState.decision = [BET_OPTIONS.indexOf(this.playerDecision), null];
                } else {
                    copyOfGameState.decision = this.playerDecision; // This should be the index of the decision already
                }
                
                jsonData = await this.postToAPI('/api/play-action', copyOfGameState);
            } else {
                copyOfGameState.declared_trump = SUIT_OPTIONS.indexOf(this.playerDecision);
                jsonData = await this.postToAPI('/api/declare-trump', copyOfGameState);
            }
            
            // Update the game state based on the response from the API after playing the action.
            this.lastBettingGameState = jsonData;
            if (typeof this.playerDecision === "string") {
                this.players[this.currentPlayer].setBet(this.playerDecision); // Update the player's bet based on their decision
            } else if (this.playerDecision[0] !== null) {
                this.players[this.currentPlayer].setBet(BET_OPTIONS[this.playerDecision[0]]); // Update the player's bet based on their decision index
            } else {
                // We sent a dummy action to compile the results after the main game, so we don't want to update the bet display in this case.
                await this.startNewRound(); // Start a new round with the updated game state from the API, which will reset the bets and update the scores.
            }
            this.bet = jsonData.bet;
            this.currentPlayer = jsonData.current_player;
            this.bettingPlayer = jsonData.betting_player;
            this.bettingPhase = jsonData.main_game === null; // If main_game is null, we're still in the betting phase
            this.playerDecision = null; // Reset the user's decision for the next turn
            this.waitingOnPlayer = false; // Reset this to indicate we've processed the user's decision
            this.readyForNextStep = true; // Set this to true to allow the game loop to proceed to the next step after processing the user's decision
            if (!this.bettingPhase) {
                // Betting phase is over. Update the players' hands with the updated game state from the API, which may include information about the declared trump suit and any cards that were revealed during betting.
                this.trumpSuit = this.lastBettingGameState.main_game.trump;
                let shift = 4 - this.bettingPlayer;
                this.trumpsAtStart = this.lastBettingGameState.main_game.trumps_at_start.slice(shift).concat(this.lastBettingGameState.main_game.trumps_at_start.slice(0, shift));
                for (let i = 0; i < this.players.length; i++) {
                    this.players[i].setCards(this.lastBettingGameState.main_game.hands[(shift + i) % 4]);
                    this.players[i].setBet(null); // Clear the bet display for all players now that we're moving to the main game phase
                }
                this.currentPlayer = (this.bettingPlayer + this.lastBettingGameState.main_game.current_player) % 4;
            }
            if (this.lastBettingGameState.done) {
                // The game is finished, we can display some end-of-game text or options here in the future.
                this.done = true;
            }
        } catch (error) {
            console.error("Failed to play action:", error);
        }
    }

    async mainPhaseUpdate() {
        try {
            let copyOfGameState = JSON.parse(JSON.stringify(this.lastBettingGameState)); // Create a deep copy of the last betting game state
            // If player decision is a string (e.g. "Pass", "Shoot"), we need to convert it to the corresponding index that the API expects.
            if (typeof this.playerDecision === "string") {
                copyOfGameState.main_game.decision = [SUIT_OPTIONS_2.indexOf(this.playerDecision.at(-1)) * 13 + RANK_OPTIONS.indexOf(this.playerDecision.slice(0, -1))];
                this.players[this.currentPlayer].setDisplayCard([this.playerDecision.at(-1), this.playerDecision.slice(0, -1)]); // Update the player's display card based on their decision
            } else {
                copyOfGameState.main_game.decision = this.playerDecision; // This should be the index of the decision already
                this.players[this.currentPlayer].setDisplayCard([SUIT_OPTIONS_2[Math.floor(this.playerDecision[0] / 13)], RANK_OPTIONS[this.playerDecision[0] % 13]]); // Update the player's display card based on their decision index
            }
            
            let jsonData = await this.postToAPI('/api/play-action', copyOfGameState);

            // Update the game state based on the response from the API after playing the action.
            this.lastBettingGameState = jsonData;
            let shift = 4 - this.bettingPlayer;
            for (let i = 0; i < this.players.length; i++) {
                this.players[i].setCards(this.lastBettingGameState.main_game.hands[(shift + i) % 4]);
            }

            if (this.lastBettingGameState.main_game.trick.length === 0 && this.lastBettingGameState.main_game.card_num > 0) {
                // Pause a bit between rounds.
                await new Promise(resolve => setTimeout(resolve, 3000));
                this.players.forEach(player => player.setDisplayCard(null)); // Clear the display card for all players at the end of the trick
            }

            this.currentPlayer = (this.bettingPlayer + this.lastBettingGameState.main_game.current_player) % 4;
            this.playerDecision = null; // Reset the user's decision for the next turn
            this.waitingOnPlayer = false;
            this.readyForNextStep = true; // Set this to true to allow the game loop to proceed to the next step after processing the user's decision
            if (this.lastBettingGameState.main_game.done) {
                this.bettingPhase = true;
                this.playerDecision = [null, null]; // Reset the player's decision for the next round
                this.message = `Betting team got ${this.lastBettingGameState.main_game.bet_points[0]} points.\nOther team got ${this.lastBettingGameState.main_game.bet_points[1]} points.`;
                // Wait for 4 seconds to let the user read the message before starting the next round.
                await new Promise(resolve => setTimeout(resolve, 4000));
                this.message = "";
            }
        } catch (error) {
            console.error("Failed to play action:", error);
        }
    }

    async chimeraAction() {
        try {
            let copyOfGameState = JSON.parse(JSON.stringify(this.lastBettingGameState)); // Create a deep copy of the last betting game state
            let jsonData = await this.postToAPI('/api/chimera-action', copyOfGameState);
            this.playerDecision = jsonData;
            // Add an artificial delay to make it feel like CHIMERA is "thinking"
            await new Promise(resolve => setTimeout(resolve, 1000));
        } catch (error) {
            console.error("Failed to get CHIMERA's action:", error);
        }
    }

    async update() {
        if (this.done) {
            return; // Game is finished, we can display some end-of-game text or options here in the future.
        }
        if (!this.setupComplete) {
            return; // Don't do anything until we've received the initial game state from the API
        }
        if (!this.readyForNextStep) {
            return; // Don't do anything until we're ready for the next step
        }
        this.readyForNextStep = false; // Set this to false to prevent the game loop from proceeding until we've completed the current step (e.g. waiting for API response, processing user decision, etc.)
        
        // Step logic
        if (!this.waitingOnPlayer) {
            if (this.bettingPlayer === null) { // Betting player hasn't been determined yet, we're still waiting for players to make their bets
                if (this.currentPlayer % 2 === 0) {
                    if (this.playerDecision === null) {
                        // It's the user's turn (Player 1 or Player 3) to choose.
                        this.players[this.currentPlayer].setShowCards(true); // Show the user's cards
                        let obs_json = await this.postToAPI('/api/get-obs', this.lastBettingGameState);
                        this.decisionOptions = obs_json.legal_actions.map(action => BET_OPTIONS[action]); // Map the valid action indices to the actual options             
                        this.waitingOnPlayer = true; // Set this to true to indicate we're waiting on the user to make a decision
                    } else {
                        // We have the user's decision, we can now send it to the API and update the game state accordingly.
                        this.players[this.currentPlayer].setShowCards(false); // Hide the user's cards again (we will show them again when it's their turn in the future)
                        await this.bettingPhaseUpdate();
                    }
                } else {
                    // It's CHIMERA's turn (Player 2 or Player 4) to choose.
                    await this.chimeraAction();
                    await this.bettingPhaseUpdate();
                }
            } else {
                if (this.bettingPhase) {
                    // Betting phase should be over, but the update didn't indicate that since the player 1 or player 3's bet suit is still null.
                    if (this.playerDecision === null) {
                        // It's the user's turn (Player 1 or Player 3) to declare trump suit after betting.
                        this.players[this.bettingPlayer].setShowCards(true); // Show the user's cards
                        this.decisionOptions = SUIT_OPTIONS; // The options for declaring trump suit
                        this.waitingOnPlayer = true; // Set this to true to indicate we're waiting on the user to declare the trump suit
                    } else {
                        // We have the user's decision for the trump suit, we can now send it to the API and update the game state accordingly.
                        this.players[this.bettingPlayer].setShowCards(false);
                        await this.bettingPhaseUpdate();
                    }
                } else {
                    // In the main game!
                    if (this.currentPlayer % 2 === 0) {
                        if (this.playerDecision === null) {
                            // It's the user's turn (Player 1 or Player 3) to play a card.
                            this.players[this.currentPlayer].setShowCards(true); // Show the user's cards
                            let obs_json = await this.postToAPI('/api/get-obs', this.lastBettingGameState);
                            this.decisionOptions = obs_json.legal_actions.map(action => `${RANK_OPTIONS[action % 13]}${SUIT_OPTIONS_2[Math.floor(action / 13)]}`);
                            this.waitingOnPlayer = true; // Set this to true to indicate we're waiting on the user to play a card
                        } else {
                            // We have the user's decision, we can now send it to the API and update the game state accordingly.
                            this.players[this.currentPlayer].setShowCards(false); // Hide the user's cards again (we will show them again when it's their turn in the future)
                            await this.mainPhaseUpdate();
                        }
                    } else {
                        // It's CHIMERA's turn (Player 2 or Player 4) to play a card.
                        await this.chimeraAction();
                        await this.mainPhaseUpdate();
                    }
                }
            }
        }

        if (this.waitingOnPlayer && decisionDiv.innerHTML === "") { // Waiting on the user to make a decision, but we haven't rendered the buttons yet
            // Display the decisions that can be clicked.
            // The buttons can be attached to the DOM and positioned over the canvas.
            for (let option of this.decisionOptions) {
                let button = document.createElement("button");
                button.innerText = option; // In the future, we can format this to be more user-friendly
                button.classList.add("btn", "btn-primary", "m-1");
                button.onclick = () => {
                    this.playerDecision = option;
                    this.waitingOnPlayer = false; // Reset this to indicate we've received the user's decision
                    this.readyForNextStep = true; // Set this to true to allow the game loop to proceed to the next step after processing the user's decision
                    decisionDiv.innerHTML = ""; // Clear the decision buttons
                }
                decisionDiv.appendChild(button);

                // Move the decision div on top of the canvas and center it (this is a simple approach, we can improve the styling later)
                decisionDiv.style.position = "absolute";
                decisionDiv.style.top = "50%";
                decisionDiv.style.left = "50%";
                decisionDiv.style.transform = "translate(-50%, -50%)";
            }
        }
    }

    async postToAPI(endpoint, payload) {
        try {
            // Send the payload as base64 encoded JSON string to the API endpoint
            let payload_encoded = btoa(JSON.stringify(payload));
            let response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: payload_encoded
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            let encodedData = await response.text();
            const decodedData = atob(encodedData);
            const jsonData = JSON.parse(decodedData);
            return jsonData;

        } catch (error) {
            console.error("Error calling API:", error);
            throw error; // Re-throw so the calling method knows something went wrong
        }
    }

    display() {
        // Display all players and their cards
        for (let player of players) {
            player.display();
        }

        // Display scores, round number, and betting player in the upper right corner
        fill(255);
        textSize(16);
        textAlign(RIGHT, TOP);
        text(`Round: ${this.round}`, width - 20, 20);
        text(`Team 1: ${this.score[0]} pts`, width - 20, 40);
        text(`Team 2: ${this.score[1]} pts`, width - 20, 60);
        if (this.bettingPlayer !== null) {
            text(`Betting Player: ${this.players[this.bettingPlayer].name}`, width - 20, 80);
        }
        if (this.bet !== null) {
            text(`Current Bet: ${this.bet}`, width - 20, 100);
        }
        if (this.trumpSuit !== null) {
            text(`Trump Suit: ${this.trumpSuit}`, width - 20, 120);
        }
        if (typeof this.trumpsAtStart === "object" && this.trumpsAtStart.length > 0) {
            text(`Trumps at Start (P1-P4): ${this.trumpsAtStart.join(", ")}`, width - 20, 140);
        }
        if (this.countInWidow !== null) {
            text(`Cards in Widow: ${this.countInWidow}`, width - 20, 160);
        }

        // Display any messages (e.g. points scored in the last round) in the center of the screen
        if (this.message) {
            fill(0, 200);
            rect(0, 0, width, height);
            fill(255);
            textSize(32);
            textAlign(CENTER, CENTER);
            text(this.message, width / 2, height / 2);
        }

        // Display an overlay with the winner and final scores if the game is done
        if (this.done) {
            fill(0, 200);
            rect(0, 0, width, height);
            fill(255);
            textSize(32);
            textAlign(CENTER, CENTER);
            let winningTeam = this.lastBettingGameState.winning_team ? "Team 1 (Players 1 & 3)" : "Team 2 (Players 2 & 4)";
            text(`Game Over! ${winningTeam} wins!\nFinal Score - Team 1: ${this.score[0]} pts, Team 2: ${this.score[1]} pts\nReload the page to play again.`, width / 2, height / 2);
        }
    }
}
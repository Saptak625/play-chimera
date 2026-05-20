// Player Class
const CARD_WIDTH = 60;
const CARD_HEIGHT = 90;
const HAND_SIZE = 9;
const SUITS = { 'H': 0, 'D': 1, 'C': 2, 'S': 3 };
const RANKS = { '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14 };
const SUITS_TO_NAME = { 'H': 'hearts', 'D': 'diamonds', 'C': 'clubs', 'S': 'spades' };
const FACE_CARD_TO_NAME = { 'J': 'jack', 'Q': 'queen', 'K': 'king', 'A': 'ace' };

const path_to_card_images = "static/playing_cards/"; // Base path for card images

let cardImages = {}; // Global object to hold images

function getCardImagePath(card) {
    // Card will be represented as a tuple like ['H', 'A'] for Ace of Hearts
    let suit = SUITS_TO_NAME[card[0]];
    let rank = card[1];
    let rankName = FACE_CARD_TO_NAME[rank] || rank;
    return `${path_to_card_images}${rankName}_of_${suit}.png`;
}

function preload() {
  const suits = ['H', 'D', 'C', 'S'];
  const values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];

  // Loop through and preload all 52 cards dynamically
  for (let suit of suits) {
    for (let val of values) {
      let cardKey = `${suit}${val}`;
      cardImages[cardKey] = loadImage(getCardImagePath([suit, val]));
    }
  }
}

class Player {
	constructor(name, position, numCards) {
        this.name = name;
        this.position = position;
        this.cards = []; // Array to hold card data objects in the future
        this.displayCard = null; // Store the card to display when the player plays a card
        this.showCards = false; // Whether to show the player's cards (only true for the user)
        this.isDealer = false;
        this.bet = null; // Store the player's current bet (if any)
        
        // Fill the hand with placeholder cards (currently just standard backs)
        for (let i = 0; i < numCards; i++) {
            this.cards.push(null); // Placeholder for card data, will be replaced with actual card objects later
        }
    }

    setShowCards(show) {
        this.showCards = show;
    }

    setDealer(isDealer) {
        this.isDealer = isDealer;
    }

    setCards(cards) {
        // Sort the array of card data objects by suit and rank
        cards.sort((a, b) => {
            if (a[0] === b[0]) { // Compare the suits
                return RANKS[a[1]] - RANKS[b[1]]; // If suits are the same, compare ranks
            }
            return SUITS[a[0]] - SUITS[b[0]]; // Otherwise, sort by suit
        });
        this.cards = cards; // Replace the placeholder cards with the actual card data objects
    }

    setDisplayCard(card) {
        this.displayCard = card; // Store the card that the player has played to display it in the center
    }

    setBet(bet) {
        this.bet = bet;
    }

	display() {
		push();
		
		// Set up text styling for player names
		fill(255);
		noStroke();
		textSize(16);
		textAlign(CENTER, CENTER);

		// Dynamic layout positioning based on table side
		if (this.position === "bottom") {
			translate(width / 2, height - 80);
			text(this.name, 0, 60);
            if (this.isDealer) {
                this.drawDealerChip(0, -80);
            }
            if (this.displayCard !== null) {
                this.drawCardFace(-CARD_WIDTH / 2, -2.5 * CARD_HEIGHT, this.displayCard); // Show the card the player has played in the center
            }
            if (this.bet !== null) {
                text(`Bet: ${this.bet}`, 0, -130); // Display the player's bet above their name
            }
			this.drawHand(0);
		} 
		else if (this.position === "top") {
			translate(width / 2, 80);
			text(this.name, 0, -60);
            if (this.isDealer) {
                this.drawDealerChip(0, 80);
            }
            if (this.displayCard !== null) {
                this.drawCardFace(-CARD_WIDTH / 2, 1.5 * CARD_HEIGHT, this.displayCard); // Show the card the player has played in the center
            }
            if (this.bet !== null) {
                text(`Bet: ${this.bet}`, 0, 130); // Display the player's bet below their name
            }
			this.drawHand(0);
		} 
		else if (this.position === "left") {
			translate(80, height / 2);
			rotate(HALF_PI);
			text(this.name, 0, 60);
            if (this.isDealer) {
                this.drawDealerChip(0, -80);
            }
            if (this.displayCard !== null) {
                this.drawCardFace(-CARD_WIDTH / 2, -2.5 * CARD_HEIGHT, this.displayCard); // Show the card the player has played in the center
            }
			rotate(-HALF_PI);
            if (this.bet !== null) {
                text(`Bet: ${this.bet}`, 130, 0); // Display the player's bet below their name
            }
			this.drawHand(HALF_PI); // Rotate cards to align with the left side
		} 
		else if (this.position === "right") {
			translate(width - 80, height / 2);
			rotate(-HALF_PI);
			text(this.name, 0, 60);
            if (this.isDealer) {
                this.drawDealerChip(0, -80);
            }
            if (this.displayCard !== null) {
                this.drawCardFace(-CARD_WIDTH / 2, -2.5 * CARD_HEIGHT, this.displayCard); // Show the card the player has played in the center
            }
			rotate(HALF_PI);
            if (this.bet !== null) {
                text(`Bet: ${this.bet}`, -130, 0); // Display the player's bet below their name
            }
			this.drawHand(HALF_PI); // Rotate cards to align with the right side
		}
		
		pop();
	}

    drawHand(rotation) {
        if (this.showCards) {
            this.drawShownHand(rotation);
        } else {
            this.drawHiddenHand(rotation);
        }
    }

    drawDealerChip(posx, posy) {
        fill(255, 215, 0);
        noStroke();
        ellipse(posx, posy, 30, 30);
        fill(0);
        textSize(15);
        textAlign(CENTER, CENTER);
        text("D", posx, posy);
        fill(255);
    }

    drawShownHand(rotation) {
        push();
        rotate(rotation);
        let spacing = 30; // Spacing between cards
        let totalWidth = (this.cards.length - 1) * spacing + CARD_WIDTH;
        let startX = -totalWidth / 2;

        for (let i = 0; i < this.cards.length; i++) {
            let card = this.cards[i];
            if (card) {
                let x = startX + i * spacing;
                let y = -CARD_HEIGHT / 2;
                this.drawCardFace(x, y, card);
            }
        }
        pop();
    }

    drawCardFace(x, y, card) {
        let cardKey = `${card[0]}${card[1]}`;
        image(cardImages[cardKey], x, y, CARD_WIDTH, CARD_HEIGHT);
    }

	drawHiddenHand(rotation) {
		push();
		rotate(rotation);
		
		let spacing = 15; // Overlap spacing between cards
		// Calculate total width of the hand to center it perfectly
		let totalWidth = (this.cards.length - 1) * spacing + CARD_WIDTH;
		let startX = -totalWidth / 2;

		for (let i = 0; i < this.cards.length; i++) {
			let x = startX + i * spacing;
			let y = -CARD_HEIGHT / 2;
			
			this.drawCardBack(x, y);
		}
		pop();
		}

    drawCardBack(x, y) {
		stroke(0);
		strokeWeight(1);
		
		// Outer white border
		fill(255);
		rect(x, y, CARD_WIDTH, CARD_HEIGHT, 4);
		
		// Inner red card back pattern
		fill(180, 0, 0);
		rect(x + 3, y + 3, CARD_WIDTH - 6, CARD_HEIGHT - 6, 2);
		
		// Simple design pattern on the back
		stroke(255, 255, 255, 100);
		line(x + 3, y + 3, x + CARD_WIDTH - 3, y + CARD_HEIGHT - 3);
		line(x + CARD_WIDTH - 3, y + 3, x + 3, y + CARD_HEIGHT - 3);
	}
}
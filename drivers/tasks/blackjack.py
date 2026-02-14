"""
PBAI Blackjack - Play manually or watch PBAI play

PBAI uses card counting, manifold memory, thermal decisions.

Run:
    python drivers/tasks/blackjack.py
    python drivers/tasks/blackjack.py --no-llm
"""

import tkinter as tk
from tkinter import messagebox
import random
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def value(self):
        if self.rank in ['J', 'Q', 'K']:
            return 10
        elif self.rank == 'A':
            return 11
        return int(self.rank)


class Deck:
    suits = ['♠', '♥', '♦', '♣']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    def __init__(self, num_decks=6):
        self.num_decks = num_decks
        self.cards = []
        self.rebuild()

    def rebuild(self):
        self.cards = []
        for _ in range(self.num_decks):
            for suit in self.suits:
                for rank in self.ranks:
                    self.cards.append(Card(suit, rank))
        random.shuffle(self.cards)

    def deal(self):
        if len(self.cards) < 20:
            self.rebuild()
        return self.cards.pop()

    def remaining(self):
        return len(self.cards)


class Hand:
    def __init__(self):
        self.cards = []
        self.bet = 0
        self.is_doubled = False
        self.is_finished = False
        self.last_action = None
        self.last_situation_key = None
        self.last_state = None
        self.outcome_recorded = False

    def add_card(self, card):
        self.cards.append(card)

    def get_value(self):
        value = sum(c.value() for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == 'A')
        while value > 21 and aces:
            value -= 10
            aces -= 1
        return value

    def is_soft(self):
        value = sum(c.value() for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == 'A')
        while value > 21 and aces:
            value -= 10
            aces -= 1
        return aces > 0 and value <= 21

    def is_blackjack(self):
        return len(self.cards) == 2 and self.get_value() == 21

    def is_busted(self):
        return self.get_value() > 21

    def can_split(self):
        return len(self.cards) == 2 and self.cards[0].value() == self.cards[1].value()

    def can_double(self):
        return len(self.cards) == 2


class BlackjackApp:
    def __init__(self, root, use_llm=True):
        self.root = root
        self.root.title("PBAI Blackjack")
        self.root.geometry("950x600")
        self.root.configure(bg='#0B6623')

        self.deck = Deck(6)
        self.player_hands = []
        self.current_hand_index = 0
        self.dealer_hand = Hand()
        self.balance = 1000
        self.starting_balance = 1000
        self.current_bet = 10
        self.game_active = False

        self.pbai_on = False
        self._pbai_deal_scheduled = False  # prevent duplicate deal chains
        self.driver = None
        self.env_core = None
        self.use_llm = use_llm
        self.delay = 800
        self.last_state = None
        self.last_action = None
        self.last_situation_key = None

        self.setup_ui()

    def setup_ui(self):
        main = tk.Frame(self.root, bg='#0B6623')
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left - game
        game = tk.Frame(main, bg='#0B6623')
        game.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(game, text="PBAI BLACKJACK", font=('Arial', 18, 'bold'),
                 bg='#0B6623', fg='gold').pack(pady=3)

        self.balance_label = tk.Label(game, text=f"${self.balance}",
                                      font=('Arial', 14, 'bold'), bg='#0B6623', fg='white')
        self.balance_label.pack()

        # Dealer
        tk.Label(game, text="DEALER", font=('Arial', 11, 'bold'), bg='#0B6623', fg='white').pack(pady=(10, 0))
        self.dealer_cards = tk.Label(game, text="", font=('Arial', 18), bg='#0B6623', fg='white')
        self.dealer_cards.pack()
        self.dealer_value = tk.Label(game, text="", font=('Arial', 10), bg='#0B6623', fg='yellow')
        self.dealer_value.pack()

        # Player
        tk.Label(game, text="PLAYER", font=('Arial', 11, 'bold'), bg='#0B6623', fg='white').pack(pady=(10, 0))
        self.player_cards = tk.Label(game, text="", font=('Arial', 18), bg='#0B6623', fg='white')
        self.player_cards.pack()
        self.player_value = tk.Label(game, text="", font=('Arial', 10), bg='#0B6623', fg='yellow')
        self.player_value.pack()

        # Bets
        bets = tk.Frame(game, bg='#0B6623')
        bets.pack(pady=5)
        for amt in [10, 25, 50, 100]:
            tk.Button(bets, text=f"${amt}", command=lambda a=amt: self.set_bet(a),
                      font=('Arial', 8), width=4).pack(side=tk.LEFT, padx=2)

        # Actions
        acts = tk.Frame(game, bg='#0B6623')
        acts.pack(pady=5)

        self.deal_btn = tk.Button(acts, text="DEAL", command=self.deal_cards,
                                  font=('Arial', 9, 'bold'), width=7, bg='#4CAF50', fg='white')
        self.deal_btn.grid(row=0, column=0, padx=2, pady=2)

        self.hit_btn = tk.Button(acts, text="HIT", command=self.hit,
                                 font=('Arial', 9, 'bold'), width=7, bg='#2196F3', fg='white', state=tk.DISABLED)
        self.hit_btn.grid(row=0, column=1, padx=2, pady=2)

        self.stand_btn = tk.Button(acts, text="STAND", command=self.stand,
                                   font=('Arial', 9, 'bold'), width=7, bg='#FF9800', fg='white', state=tk.DISABLED)
        self.stand_btn.grid(row=0, column=2, padx=2, pady=2)

        self.double_btn = tk.Button(acts, text="DOUBLE", command=self.double_down,
                                    font=('Arial', 9, 'bold'), width=7, bg='#9C27B0', fg='white', state=tk.DISABLED)
        self.double_btn.grid(row=1, column=0, padx=2, pady=2)

        self.split_btn = tk.Button(acts, text="SPLIT", command=self.split,
                                   font=('Arial', 9, 'bold'), width=7, bg='#FF5722', fg='white', state=tk.DISABLED)
        self.split_btn.grid(row=1, column=1, padx=2, pady=2)

        # PBAI
        pbai = tk.Frame(game, bg='#0B6623')
        pbai.pack(pady=5)

        self.pbai_btn = tk.Button(pbai, text="▶ PBAI", font=('Arial', 9, 'bold'),
                                   bg='#22c55e', fg='white', width=10, command=self.toggle_pbai)
        self.pbai_btn.pack(side=tk.LEFT, padx=3)

        tk.Label(pbai, text="Delay:", bg='#0B6623', fg='white', font=('Arial', 8)).pack(side=tk.LEFT)
        self.delay_var = tk.StringVar(value="800")
        tk.Entry(pbai, textvariable=self.delay_var, width=4, font=('Arial', 8)).pack(side=tk.LEFT)

        self.message = tk.Label(game, text="Bet then DEAL, or PBAI Auto",
                                font=('Arial', 10, 'bold'), bg='#0B6623', fg='yellow')
        self.message.pack(pady=5)

        # Right - nodes
        nodes = tk.Frame(main, bg='#1a1a2e', width=300)
        nodes.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        nodes.pack_propagate(False)

        tk.Label(nodes, text="PBAI State", font=('Arial', 11, 'bold'),
                 bg='#1a1a2e', fg='#00ff88').pack(pady=3)

        self.count_lbl = tk.Label(nodes, text="RC: 0 | TC: 0.0", font=('Arial', 9), bg='#1a1a2e', fg='#88ff88')
        self.count_lbl.pack()

        self.stats_lbl = tk.Label(nodes, text="W: 0 | L: 0 | P: $0", font=('Arial', 9), bg='#1a1a2e', fg='#88ff88')
        self.stats_lbl.pack()

        self.weight_lbl = tk.Label(nodes, text="Cons: 0.50 | Bet: 0.50", font=('Arial', 9), bg='#1a1a2e', fg='#88ff88')
        self.weight_lbl.pack()

        tk.Label(nodes, text="Hot Nodes:", font=('Arial', 9, 'bold'), bg='#1a1a2e', fg='#00ff88').pack(pady=(8, 2))

        self.nodes_text = tk.Text(nodes, font=('Courier', 7), bg='#0a0a1a', fg='#00ff88', width=35, height=20)
        self.nodes_text.pack(padx=3, pady=3, fill=tk.BOTH, expand=True)

    def set_bet(self, amt):
        if not self.game_active and not self.pbai_on:
            self.current_bet = amt

    def update_nodes(self):
        if not self.driver:
            return
        stats = self.driver.get_stats()
        self.count_lbl.config(text=f"RC: {stats['running_count']} | TC: {stats['true_count']:.1f}")
        self.stats_lbl.config(text=f"W:{stats['hands_won']} L:{stats['hands_lost']} T:{stats['hands_pushed']} | ${stats['profit']}")
        self.weight_lbl.config(text=f"Cons: {stats['conservation_weight']:.2f} | Bet: {stats.get('bet_weight', 0.5):.2f}")

        self.nodes_text.delete(1.0, tk.END)
        nodes = sorted(self.driver.manifold.nodes.values(), key=lambda n: n.heat, reverse=True)
        for n in nodes[:25]:
            if n.concept not in ['blackjack'] and not n.concept.startswith('bootstrap'):
                bar = '█' * min(8, int(n.heat))
                self.nodes_text.insert(tk.END, f"{n.concept[:22]}\n H={n.heat:.1f} {bar}\n")

    def deal_cards(self):
        if self.current_bet > self.balance:
            self.current_bet = max(10, self.balance)
        if self.current_bet > self.balance:
            if not self.pbai_on:
                messagebox.showerror("Error", "No money!")
            return

        self.balance -= self.current_bet
        self.game_active = True

        self.player_hands = [Hand()]
        self.player_hands[0].bet = self.current_bet
        self.current_hand_index = 0
        self.dealer_hand = Hand()

        for _ in range(2):
            c = self.deck.deal()
            self.player_hands[0].add_card(c)
            if self.driver:
                self.driver.count_card(c.rank)

        for _ in range(2):
            c = self.deck.deal()
            self.dealer_hand.add_card(c)
            if self.driver:
                self.driver.count_card(c.rank)

        self.update_display(True)
        self.update_buttons()

        if self.player_hands[0].is_blackjack():
            if self.dealer_hand.is_blackjack():
                self.balance += self.current_bet
                self.message.config(text="Push - both BJ")
                if self.driver:
                    result = self.driver.record_blackjack(won=False, amount=self.current_bet)
                    if self.env_core:
                        self.env_core.feedback(result)
            else:
                win = int(self.current_bet * 2.5)
                self.balance += win
                self.message.config(text=f"Blackjack! +${win - self.current_bet}")
                if self.driver:
                    result = self.driver.record_blackjack(won=True, amount=self.current_bet)
                    if self.env_core:
                        self.env_core.feedback(result)
            self.reset_game()
            self.update_nodes()
            if self.pbai_on:
                self._schedule_pbai_deal()
            return

        self.message.config(text="Hit, Stand, Double, Split?")
        self.last_state = self._get_state()
        self.update_nodes()

    def _get_state(self):
        if not self.player_hands or self.current_hand_index >= len(self.player_hands):
            return None
        h = self.player_hands[self.current_hand_index]
        from drivers.blackjack_driver import HandState
        return HandState(
            player_value=h.get_value(),
            dealer_upcard=self.dealer_hand.cards[0].value(),
            is_soft=h.is_soft(),
            can_double=h.can_double() and h.bet <= self.balance,
            can_split=h.can_split() and h.bet <= self.balance,
            pair_value=h.cards[0].value() if h.can_split() else None,
            num_cards=len(h.cards)
        )

    def hit(self):
        if not self.player_hands:
            return
        h = self.player_hands[self.current_hand_index]

        # Capture situation BEFORE card is dealt (with current count)
        if self.driver and self.last_state:
            self.last_situation_key = self.driver.record_decision(self.last_state, "hit")
            self.last_action = "hit"
            h.last_action = "hit"
            h.last_situation_key = self.last_situation_key
            h.last_state = self.last_state

        c = self.deck.deal()
        h.add_card(c)
        if self.driver:
            self.driver.count_card(c.rank)

        self.update_display(True)
        self.double_btn.config(state=tk.DISABLED)
        self.split_btn.config(state=tk.DISABLED)

        if h.is_busted():
            h.is_finished = True
            self.message.config(text="Bust!")
            if self.driver and h.last_situation_key:
                result = self.driver.record_outcome(h.last_state, "hit", False, h.bet,
                                                    situation_key=h.last_situation_key)
                h.outcome_recorded = True
                if self.env_core:
                    self.env_core.feedback(result)
            self.next_hand()
        else:
            self.last_state = self._get_state()
        self.update_nodes()

    def stand(self):
        if not self.player_hands or self.current_hand_index >= len(self.player_hands):
            return
        h = self.player_hands[self.current_hand_index]
        if self.driver and self.last_state:
            self.last_situation_key = self.driver.record_decision(self.last_state, "stand")
            self.last_action = "stand"
            h.last_action = "stand"
            h.last_situation_key = self.last_situation_key
            h.last_state = self.last_state
        h.is_finished = True
        self.next_hand()
        self.update_nodes()

    def double_down(self):
        if not self.player_hands:
            return
        h = self.player_hands[self.current_hand_index]
        if h.bet > self.balance:
            return

        self.balance -= h.bet
        h.bet *= 2
        h.is_doubled = True

        # Capture situation BEFORE card is dealt
        if self.driver and self.last_state:
            self.last_situation_key = self.driver.record_decision(self.last_state, "double")
            self.last_action = "double"
            h.last_action = "double"
            h.last_situation_key = self.last_situation_key
            h.last_state = self.last_state

        c = self.deck.deal()
        h.add_card(c)
        if self.driver:
            self.driver.count_card(c.rank)

        self.update_display(True)
        h.is_finished = True

        if h.is_busted():
            self.message.config(text="Bust on double!")
            if self.driver and h.last_situation_key:
                result = self.driver.record_outcome(h.last_state, "double", False, h.bet,
                                                    situation_key=h.last_situation_key)
                h.outcome_recorded = True
                if self.env_core:
                    self.env_core.feedback(result)

        self.next_hand()
        self.update_nodes()

    def split(self):
        if not self.player_hands:
            return
        h = self.player_hands[self.current_hand_index]
        if not h.can_split():
            return
        if h.bet > self.balance:
            return

        # Capture situation BEFORE cards are dealt
        if self.driver and self.last_state:
            self.last_situation_key = self.driver.record_decision(self.last_state, "split")
            self.last_action = "split"

        self.balance -= h.bet
        new_h = Hand()
        new_h.bet = h.bet
        new_h.add_card(h.cards.pop())

        c1, c2 = self.deck.deal(), self.deck.deal()
        h.add_card(c1)
        new_h.add_card(c2)

        if self.driver:
            self.driver.count_card(c1.rank)
            self.driver.count_card(c2.rank)

        self.player_hands.insert(self.current_hand_index + 1, new_h)
        self.update_display(True)
        self.update_buttons()
        self.last_state = self._get_state()
        self.update_nodes()

    def next_hand(self):
        self.current_hand_index += 1
        if self.current_hand_index < len(self.player_hands):
            self.update_display(True)
            self.update_buttons()
            self.message.config(text=f"Hand {self.current_hand_index + 1}")
            self.last_state = self._get_state()
        else:
            self.dealer_play()

    def dealer_play(self):
        self.update_display(False)
        if all(h.is_busted() for h in self.player_hands):
            self.finish_game()
            return
        self.root.after(400, self.dealer_draw)

    def dealer_draw(self):
        if self.dealer_hand.get_value() < 17:
            c = self.deck.deal()
            self.dealer_hand.add_card(c)
            if self.driver:
                self.driver.count_card(c.rank)
            self.update_display(False)
            self.root.after(400, self.dealer_draw)
        else:
            self.finish_game()

    def finish_game(self):
        dv = self.dealer_hand.get_value()
        results = []
        total = 0

        for i, h in enumerate(self.player_hands):
            pv = h.get_value()
            won = False
            push = False

            if h.is_busted():
                results.append(f"H{i + 1}: Lost")
            elif dv > 21 or pv > dv:
                total += h.bet * 2
                results.append(f"H{i + 1}: +${h.bet}")
                won = True
            elif pv < dv:
                results.append(f"H{i + 1}: Lost")
            else:
                total += h.bet
                results.append(f"H{i + 1}: Push")
                push = True

            if self.driver and h.last_action and not h.outcome_recorded:
                if push:
                    result = self.driver.record_push(h.last_state, h.last_action, h.bet,
                                                     situation_key=h.last_situation_key)
                    h.outcome_recorded = True
                    if self.env_core:
                        self.env_core.feedback(result)
                elif not h.is_busted():
                    result = self.driver.record_outcome(h.last_state, h.last_action, won, h.bet,
                                                        situation_key=h.last_situation_key)
                    h.outcome_recorded = True
                    if self.env_core:
                        self.env_core.feedback(result)

        self.balance += total
        self.balance_label.config(text=f"${self.balance}")
        self.message.config(text=" | ".join(results))

        if self.driver:
            self.driver.adjust_weights(self.balance, self.starting_balance)

        if self.balance <= 0:
            if self.pbai_on:
                # Auto-reset in PBAI mode (no modal dialog)
                logger.info("BROKE — auto-resetting to $1000")
                self.message.config(text="BROKE — resetting...")
            else:
                messagebox.showinfo("Broke", "Out of money!")
            self.balance = 1000
            self.starting_balance = 1000

        self.reset_game()
        self.update_nodes()

        # Save to unified growth map
        self.save_growth()

        # Check reshuffle
        if self.deck.remaining() < 52 and self.driver:
            self.driver.reset_count()

        # Continue PBAI
        if self.pbai_on:
            self._schedule_pbai_deal()

    def reset_game(self):
        self.game_active = False
        self.last_state = None
        self.last_action = None
        self.last_situation_key = None
        self.deal_btn.config(state=tk.NORMAL)
        self.hit_btn.config(state=tk.DISABLED)
        self.stand_btn.config(state=tk.DISABLED)
        self.double_btn.config(state=tk.DISABLED)
        self.split_btn.config(state=tk.DISABLED)

    def update_buttons(self):
        if not self.game_active:
            return
        self.deal_btn.config(state=tk.DISABLED)
        self.hit_btn.config(state=tk.NORMAL)
        self.stand_btn.config(state=tk.NORMAL)
        h = self.player_hands[self.current_hand_index]
        self.double_btn.config(state=tk.NORMAL if h.can_double() and h.bet <= self.balance else tk.DISABLED)
        self.split_btn.config(state=tk.NORMAL if h.can_split() and h.bet <= self.balance else tk.DISABLED)

    def update_display(self, hide_dealer):
        if hide_dealer and len(self.dealer_hand.cards) >= 2:
            self.dealer_cards.config(text=f"{self.dealer_hand.cards[0]}  🂠")
            self.dealer_value.config(text="")
        else:
            self.dealer_cards.config(text="  ".join(str(c) for c in self.dealer_hand.cards))
            self.dealer_value.config(text=f"Value: {self.dealer_hand.get_value()}")

        if len(self.player_hands) == 1:
            self.player_cards.config(text="  ".join(str(c) for c in self.player_hands[0].cards))
            self.player_value.config(text=f"Value: {self.player_hands[0].get_value()}")
        else:
            txt = []
            for i, h in enumerate(self.player_hands):
                m = " ◄" if i == self.current_hand_index else ""
                txt.append(f"H{i + 1}{m}: {'  '.join(str(c) for c in h.cards)}")
            self.player_cards.config(text="\n".join(txt))
            idx = min(self.current_hand_index, len(self.player_hands) - 1)
            self.player_value.config(text=f"H{idx + 1}: {self.player_hands[idx].get_value()}")

        self.balance_label.config(text=f"${self.balance}")

    # ══════════════════════════════════════════════════════════════════════════
    # PBAI
    # ══════════════════════════════════════════════════════════════════════════

    def init_driver(self):
        if self.driver:
            return
        from core import get_pbai_manifold, get_growth_path
        from drivers.blackjack_driver import BlackjackDriver
        from drivers.environment import EnvironmentCore

        self.message.config(text="Loading PBAI...")
        self.root.update()

        # Get the ONE PBAI manifold (loads existing or births on first run)
        self.growth_path = get_growth_path("growth_map.json")
        manifold = get_pbai_manifold(self.growth_path)
        self.message.config(text=f"Loaded {len(manifold.nodes)} nodes")

        self.driver = BlackjackDriver(manifold=manifold)
        self.env_core = EnvironmentCore(manifold=manifold)
        self.env_core.register_driver(self.driver)
        self.env_core.activate_driver("blackjack")

        self.message.config(text="PBAI ready")
        self.update_nodes()

    def save_growth(self):
        """Save to unified growth map."""
        if self.driver and hasattr(self, 'growth_path'):
            self.driver.manifold.save_growth_map(self.growth_path)

    def toggle_pbai(self):
        if self.pbai_on:
            self.stop_pbai()
        else:
            self.start_pbai()

    def start_pbai(self):
        self.init_driver()
        self.pbai_on = True
        self.pbai_btn.config(text="⏹ Stop", bg="#ef4444")
        self.message.config(text="PBAI playing...")
        self.pbai_deal()

    def stop_pbai(self):
        self.pbai_on = False
        self._pbai_deal_scheduled = False
        self.pbai_btn.config(text="▶ PBAI", bg="#22c55e")

    def _schedule_pbai_deal(self, delay=1500):
        """Schedule a single pbai_deal callback, preventing duplicates."""
        if not self._pbai_deal_scheduled:
            self._pbai_deal_scheduled = True
            self.root.after(delay, self.pbai_deal)

    def pbai_deal(self):
        self._pbai_deal_scheduled = False
        if not self.pbai_on:
            return
        if self.game_active:
            # Previous game still resolving, retry later
            self._schedule_pbai_deal(delay=500)
            return
        self.current_bet = self.driver.get_bet_size(self.balance)
        self.deal_cards()
        if self.game_active:
            try:
                d = int(self.delay_var.get())
            except:
                d = 800
            self.root.after(d, self.pbai_action)

    def pbai_action(self):
        if not self.pbai_on or not self.game_active:
            return

        state = self._get_state()
        if not state:
            self.next_hand()
            return

        # Route through EnvironmentCore: set_hand_state → perceive → decide
        self.driver.set_hand_state(state, self.balance)
        perception = self.env_core.perceive()
        action_obj = self.env_core.decide(perception)
        action = action_obj.action_type  # "hit", "stand", "double", "split"

        self.message.config(text=f"PBAI: {action.upper()}")

        try:
            d = int(self.delay_var.get())
        except:
            d = 800

        if action == "hit":
            self.hit()
            if self.game_active:
                self.root.after(d, self.pbai_action)
        elif action == "stand":
            self.stand()
            self.root.after(d * 2, self.pbai_next)
        elif action == "double":
            self.double_down()
            self.root.after(d * 2, self.pbai_next)
        elif action == "split":
            self.split()
            self.root.after(d, self.pbai_action)

        self.update_nodes()

    def pbai_next(self):
        """Continue to next action if game still active."""
        if not self.pbai_on:
            return
        if self.game_active:
            self.root.after(1, self.pbai_action)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--pbai", action="store_true", help="Auto-start PBAI mode")
    args = parser.parse_args()

    root = tk.Tk()
    app = BlackjackApp(root, use_llm=not args.no_llm)
    if args.pbai:
        root.after(500, app.start_pbai)
    root.mainloop()

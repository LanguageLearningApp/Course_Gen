import tkinter as tk  # For creating the GUI
from tkinter import messagebox  # For showing completion messages and prompts
import random  # For shuffling vocab and selecting components
import math
from components import components, sections  # Import components and sections from components.py

# =============================================================================
# Configurable Parameters
# =============================================================================

# =============================================================================
# Configurable Parameters
# =============================================================================

# CHUNK_SIZE_MAP:
# - Maps user learning speed ("slow", "average", "fast") to the number of words per chunk.
# - Increasing a chunk size means that more words will be introduced per chunk, which may speed up progress
#   but could overwhelm users with too many new words at once.
# - Decreasing the chunk size results in fewer words per chunk, leading to a slower but potentially more focused learning pace.
CHUNK_SIZE_MAP = {"slow": 3, "average": 5, "fast": 8}

# LEVELS_PER_CHUNK:
# - Defines how many levels (or iterations) are present within each chunk.
# - Increasing this value means users will have to complete more levels per chunk before moving on,
#   which can provide more practice with the current vocabulary.
# - Decreasing this value results in quicker transitions between chunks.
LEVELS_PER_CHUNK = 10

# FLEXIBLE_WORDS_MIN:
# - For components that use a "Flexible" word count, this is the minimum number of words to display.
# - Increasing this value means more words will be shown, which can help with exposure but might reduce focus.
# - Decreasing it will show fewer words, potentially making the component easier and quicker.
FLEXIBLE_WORDS_MIN = 10

# BONUS_FOR_CURRENT_CHUNK:
# - Adds a bonus to the weight of words that belong to the current chunk.
# - A higher bonus will prioritize current chunk words more strongly over older words.
# - A lower bonus reduces the emphasis on current chunk words, leading to a more balanced mix.
BONUS_FOR_CURRENT_CHUNK = 0.5

# REVIEW_DECAY_DIVISOR:
# - Controls the rate at which the review boost decays for words from previous chunks.
# - A smaller value causes a faster decay (i.e., older words lose their review boost quicker).
# - A larger value slows the decay, keeping the review boost for older words active for a longer period.
REVIEW_DECAY_DIVISOR = 2.0

# MAX_REVIEW_BOOST:
# - Sets the maximum additional weight a word can receive as a review boost.
# - Increasing this value allows a larger boost for words that were answered incorrectly.
# - Decreasing it will limit how much extra weight can be given, making the review effect more subtle.
MAX_REVIEW_BOOST = 8.0

# REVIEW_BOOST_MULTIPLIER:
# - Multiplies the difference between the word’s current weight and the baseline (1.0) when computing the review boost.
# - A higher multiplier amplifies the impact of past errors on word selection.
# - A lower multiplier softens the effect of previous performance on word weighting.
REVIEW_BOOST_MULTIPLIER = 2.5

# BASE_OTHER_PROPORTION:
# - Base proportion of words to select from previous chunks ("other" words) relative to the total word count.
# - Increasing this means more older words are likely to be included, reinforcing review.
# - Decreasing it favors selecting more new words from the current chunk.
BASE_OTHER_PROPORTION = 0.2

# OTHER_PROPORTION_SCALE:
# - Determines how much the average weight of older words influences the proportion of "other" words selected.
# - A higher scale will increase the proportion of review words if their weights are high.
# - A lower scale reduces the impact of past performance on the proportion calculation.
OTHER_PROPORTION_SCALE = 0.8

# OTHER_WEIGHT_DIVISOR:
# - Used to scale the average weight of older words when calculating their influence.
# - Increasing this divisor will decrease the impact of weight differences on the "other" word proportion.
# - Decreasing it amplifies the effect of weight variations.
OTHER_WEIGHT_DIVISOR = 4

# SECTION_CHUNK_DIVISOR:
# - Used to group words by section when applying a recency bias.
# - A larger divisor will smooth out differences across sections, while a smaller one will make section grouping more sensitive.
SECTION_CHUNK_DIVISOR = 3

# RECENCY_BIAS_THRESHOLD:
# - Sets the threshold (in chunk distance) up to which there is no recency penalty.
# - Increasing this threshold means that words from further back will still be treated as recent.
# - Decreasing it will start penalizing older chunks sooner.
RECENCY_BIAS_THRESHOLD = 9

# RECENCY_DECAY_DIVISOR:
# - Controls how quickly the recency factor decays for words beyond the threshold.
# - A larger divisor slows the decay, giving older words more chance to be selected.
# - A smaller divisor accelerates the decay, reducing the weight of older words faster.
RECENCY_DECAY_DIVISOR = 5.0

# MIN_PERF_FACTOR:
# - The minimum performance factor used when calculating selection probabilities for "other" words.
# - Increasing this value ensures that even poorly performing sections have a minimum influence.
# - Decreasing it reduces the influence of sections with low performance.
MIN_PERF_FACTOR = 0.1

# ACCURACY_WEIGHT:
# - Weight given to the accuracy component when computing a performance score.
# - A higher accuracy weight emphasizes getting words right.
# - A lower value puts relatively more emphasis on other factors like speed.
ACCURACY_WEIGHT = 0.7

# SPEED_WEIGHT:
# - Weight given to the speed component when computing a performance score.
# - Increasing this value makes pace (or speed) more influential in overall performance.
# - Decreasing it reduces the influence of speed on the performance score.
SPEED_WEIGHT = 0.3

# SPEED_MAP:
# - Maps descriptive pace ("slow", "average", "fast") to numerical values used in performance scoring.
# - Changing these numbers adjusts how much each pace contributes to the performance score.
SPEED_MAP = {"slow": 0.3, "average": 0.6, "fast": 1.0}

# WRONG_WEIGHT_INCREMENT:
# - The amount by which a word’s weight is increased when it is marked wrong.
# - Increasing this makes the algorithm more likely to review words that were answered incorrectly.
# - Decreasing it will lessen the review boost for mistakes.
WRONG_WEIGHT_INCREMENT = 0.5

# CORRECT_WEIGHT_DECREMENT:
# - The amount by which a word’s weight is decreased when it is answered correctly.
# - A higher decrement quickly reduces the review priority for mastered words.
# - A lower decrement slows the reduction, causing words to remain in the rotation longer.
CORRECT_WEIGHT_DECREMENT = 0.2

# MIN_WORD_WEIGHT:
# - The minimum weight a word can have. Ensures that a word's weight never drops below a certain level.
# - Increasing this minimum prevents words from being completely ignored.
# - Decreasing it allows for a lower bound, which could make mastered words nearly vanish from selection.
MIN_WORD_WEIGHT = 0.1

# HIGH_PERFORMANCE_THRESHOLD:
# - Threshold for a performance score above which a high performance is assumed.
# - Increasing this threshold means that only very high scores trigger “high performance” adjustments.
# - Decreasing it makes it easier to reach high performance status.
HIGH_PERFORMANCE_THRESHOLD = 0.8

# MEDIUM_PERFORMANCE_THRESHOLD:
# - Threshold for determining medium performance.
# - Adjusting this value affects how the algorithm differentiates between medium and low performance.
MEDIUM_PERFORMANCE_THRESHOLD = 0.5

# BASE_SIMULATION_ACCURACY:
# - Base probability that a word is answered correctly in the simulation.
# - Increasing this simulates a stronger overall performance.
# - Decreasing it simulates a lower baseline performance.
BASE_SIMULATION_ACCURACY = 0.7

# ADJUSTED_ACCURACY_MIN:
# - Minimum allowed accuracy after adjusting for word difficulty and past performance.
# - Increasing this raises the floor for simulated accuracy.
# - Decreasing it allows for a lower bound on accuracy, simulating more challenging conditions.
ADJUSTED_ACCURACY_MIN = 0.2

# ADJUSTED_ACCURACY_MAX:
# - Maximum allowed accuracy after adjustments.
# - Increasing this value allows for higher accuracy simulation.
# - Decreasing it limits the top performance level achievable in the simulation.
ADJUSTED_ACCURACY_MAX = 0.9

# WORD_WEIGHT_ACCURACY_DIVISOR:
# - Used to scale the influence of the word weight on the adjusted accuracy in the simulation.
# - A larger divisor reduces the impact of word weight differences on simulated accuracy.
# - A smaller divisor increases that impact, making word weights more influential on simulated outcomes.
WORD_WEIGHT_ACCURACY_DIVISOR = 10.0


# =============================================================================
# User Class and Adaptive Algorithm Implementation
# =============================================================================

class User:
    def __init__(self, learning_speed, start_from_section=0):
        # Initialize user attributes
        self.current_level = 1  # Current difficulty level (1-10) within the chunk
        self.total_levels_completed = 0  # Total number of levels completed across all chunks
        self.current_component_idx = 0  # Index of the starting component (not heavily used now)
        self.performance_history = []  # List to store performance data for each component completed

        # Store the user's self-reported learning speed (slow, average, fast)
        self.learning_speed = learning_speed
        # Set chunk size based on learning speed
        self.chunk_size = CHUNK_SIZE_MAP[learning_speed]

        # Divide sections into chunks
        self.all_chunks = []
        for section in sections:
            section_words = section["words"]
            chunks = [section_words[i:i + self.chunk_size] for i in range(0, len(section_words), self.chunk_size)]
            self.all_chunks.extend(chunks)
        self.total_chunks = len(self.all_chunks)  # Total number of chunks across all sections
        self.levels_per_chunk = LEVELS_PER_CHUNK  # Fixed number of levels per chunk
        self.max_levels = self.total_chunks * self.levels_per_chunk  # Total levels to complete all sections

        # Set starting point after completed sections
        self.start_from_section = start_from_section
        if start_from_section > 0:
            completed_chunks = sum((len(s["words"]) + self.chunk_size - 1) // self.chunk_size for s in sections[:start_from_section])
            self.total_levels_completed = completed_chunks * self.levels_per_chunk

        # Initialize active vocabulary with all words up to the current chunk
        self.active_vocab = sum(self.all_chunks[:self.total_levels_completed // self.levels_per_chunk + 1], []) if self.all_chunks else []

        # Set to track words marked wrong across all components
        self.wrong_words = set()
        # Number of correct words in the current component
        self.current_correct = 0
        # List to store the current component’s vocabulary
        self.current_vocab_list = []
        # Track words marked wrong in the current component
        self.current_wrong_words = set()

        # Initialize section performance from metadata
        self.section_performance = {section["name"]: section["performance"] for section in sections}

        # Initialize word weights for all words with default weight 1.0
        all_words = [word for section in sections for word in section["words"]]
        self.word_weights = {word: 1.0 for word in all_words}

        # Create a mapping of words to their chunk index
        self.word_to_chunk = {}
        for idx, chunk in enumerate(self.all_chunks):
            for word in chunk:
                self.word_to_chunk[word] = idx

    # Method to select vocab for the current component with weighting
    def get_current_vocab(self, component):
        # Calculate the current chunk index
        current_chunk_idx = self.total_levels_completed // self.levels_per_chunk
        if current_chunk_idx >= len(self.all_chunks):
            return []

        # Update active vocab to include all words up to the current chunk
        self.active_vocab = sum(self.all_chunks[:current_chunk_idx + 1], [])

        # Reset current_wrong_words for each new component
        self.current_wrong_words = set()

        if component["word_count"] == "N/A":
            # No words for N/A components
            selected_vocab = []
        elif component["name"] == "Flashcards" and self.total_levels_completed % self.levels_per_chunk == 0:
            # Special case: Flashcards at chunk start show only current chunk's words
            selected_vocab = self.all_chunks[current_chunk_idx]
        else:
            # Calculate effective weights for all words in active_vocab
            effective_weights = {}
            for word in self.active_vocab:
                section_perf = self.section_performance[self.get_section_for_word(word)]
                # Reduced bonus for current chunk
                bonus = BONUS_FOR_CURRENT_CHUNK if self.word_to_chunk[word] == current_chunk_idx else 0.0
                # Review boost with steeper exponential distance decay
                if self.word_to_chunk[word] < current_chunk_idx:
                    chunk_distance = current_chunk_idx - self.word_to_chunk[word]
                    distance_penalty = math.exp(-chunk_distance / REVIEW_DECAY_DIVISOR)
                    review_boost = min(MAX_REVIEW_BOOST, (self.word_weights[word] - 1) * REVIEW_BOOST_MULTIPLIER * distance_penalty)
                else:
                    review_boost = 0
                effective_weights[word] = self.word_weights[word] + bonus + review_boost

            # Determine total words to select based on component's word_count
            word_count = component["word_count"]
            if word_count == "Flexible":
                total_words = min(FLEXIBLE_WORDS_MIN, len(self.active_vocab))
            elif isinstance(word_count, int):
                total_words = word_count
            elif "-" in word_count:
                min_count, max_count = map(int, word_count.split("-"))
                total_words = random.randint(min_count, max_count)
            else:
                total_words = 0

            # Calculate "other" proportion based on average word weights of previous sections
            other_words = [w for w in self.active_vocab if self.word_to_chunk[w] < current_chunk_idx]
            if other_words:
                avg_other_weight = sum(self.word_weights[w] for w in other_words) / len(other_words)
                other_proportion = BASE_OTHER_PROPORTION + OTHER_PROPORTION_SCALE * min(1.0, (avg_other_weight - 1) / OTHER_WEIGHT_DIVISOR)
            else:
                other_proportion = BASE_OTHER_PROPORTION
            target_other = max(1, int(total_words * other_proportion))
            target_own = total_words - target_other

            # Split active_vocab into current chunk ("own") and previous ("other")
            own_words = [w for w in self.active_vocab if self.word_to_chunk[w] == current_chunk_idx]
            other_words = [w for w in self.active_vocab if self.word_to_chunk[w] < current_chunk_idx]

            # Select "own" words
            own_sorted = sorted(own_words, key=lambda w: (-(w in self.wrong_words), -effective_weights[w]))
            selected_own = own_sorted[:min(target_own, len(own_sorted))]

            # Select "other" words based on section performance with steep recency bias
            selected_other = []
            if other_words and target_other > 0:
                # Group other_words by section
                other_by_section = {}
                for word in other_words:
                    section = self.get_section_for_word(word)
                    if section not in other_by_section:
                        other_by_section[section] = []
                    other_by_section[section].append(word)

                # Calculate selection probability with steeper recency bias
                section_probs = {}
                total_prob = 0
                for section in other_by_section:
                    section_chunk_idx = max(self.word_to_chunk[w] for w in other_by_section[section]) // SECTION_CHUNK_DIVISOR
                    chunk_distance = current_chunk_idx - section_chunk_idx
                    # Steeper decay beyond threshold
                    perf_factor = max(MIN_PERF_FACTOR, 1.0 - self.section_performance[section])
                    recency_factor = math.exp(-max(0, chunk_distance - RECENCY_BIAS_THRESHOLD) / RECENCY_DECAY_DIVISOR)
                    prob = perf_factor * recency_factor
                    section_probs[section] = prob
                    total_prob += prob

                # Normalize probabilities
                if total_prob > 0:
                    for section in section_probs:
                        section_probs[section] /= total_prob

                # Select "other" words weighted by section probs
                available_sections = list(other_by_section.keys())
                while len(selected_other) < target_other and available_sections:
                    chosen_section = random.choices(available_sections, weights=[section_probs[s] for s in available_sections], k=1)[0]
                    section_words = sorted(other_by_section[chosen_section], key=lambda w: (-(w in self.wrong_words), -effective_weights[w]))
                    available = [w for w in section_words if w not in selected_other]
                    if available:
                        selected_other.append(available[0])
                    else:
                        available_sections.remove(chosen_section)
                selected_other = selected_other[:target_other]

            # Combine selections
            selected_vocab = selected_own + selected_other

            # Fill remaining slots with balanced "other" if needed
            remaining_slots = total_words - len(selected_vocab)
            if remaining_slots > 0 and other_words:
                remaining_other = [w for w in other_words if w not in selected_other]
                remaining_other_sorted = sorted(remaining_other, key=lambda w: (-(w in self.wrong_words), -effective_weights[w]))
                selected_vocab.extend(remaining_other_sorted[:remaining_slots])

            # Shuffle the final selection
            random.shuffle(selected_vocab)

        self.current_vocab_list = selected_vocab
        self.current_correct = len(selected_vocab)
        return selected_vocab

    # Method to mark a word as wrong
    def mark_word_wrong(self, word):
        self.wrong_words.add(word)
        if self.current_correct > 0:
            self.current_correct -= 1
        self.current_wrong_words.add(word)
        print(f"Marked '{word}' as wrong. Correct words: {self.current_correct}")

    # Method to update user progress and word weights
    def update_progress(self, pace, total_words, component_name):
        accuracy = self.current_correct / total_words if total_words > 0 else 1.0
        speed = SPEED_MAP[pace]
        performance_score = (accuracy * ACCURACY_WEIGHT) + (speed * SPEED_WEIGHT) if total_words > 0 else speed

        self.performance_history.append({
            "component": component_name,
            "pace": pace,
            "correct_words": self.current_correct,
            "total_words": total_words,
            "score": performance_score,
            "level": self.total_levels_completed
        })

        current_section = self.get_current_section()
        current_scores = [entry["score"] for entry in self.performance_history if self.get_section_for_level(entry["level"]) == current_section]
        self.section_performance[current_section] = sum(current_scores) / len(current_scores) if current_scores else performance_score
        print(f"Updated performance for {current_section}: {self.section_performance[current_section]:.2f}")

        # Update word weights based on component performance
        for word in self.current_vocab_list:
            if word in self.current_wrong_words:
                self.word_weights[word] += WRONG_WEIGHT_INCREMENT  # Increase weight if marked wrong
            else:
                self.word_weights[word] = max(MIN_WORD_WEIGHT, self.word_weights[word] - CORRECT_WEIGHT_DECREMENT)  # Decrease weight if correct

        recent_scores = [entry["score"] for entry in self.performance_history[-3:]] if len(self.performance_history) >= 1 else [performance_score]
        avg_recent_score = sum(recent_scores) / len(recent_scores)
        print(f"Average recent performance (last {len(recent_scores)}): {avg_recent_score:.2f}")

        # Always increment total_levels_completed to ensure progression
        self.total_levels_completed += 1
        self.current_level += 1
        if self.current_level > LEVELS_PER_CHUNK or self.total_levels_completed % self.levels_per_chunk == 0:
            self.current_level = 1
            self.wrong_words.clear()
            current_chunk_idx = self.total_levels_completed // self.levels_per_chunk
            if current_chunk_idx < self.total_chunks:
                print(f"Completed chunk {current_chunk_idx} in {self.get_current_section()}. Moving to next chunk.")
        print(f"Level advanced to {self.current_level} (Total: {self.total_levels_completed})")

    # Method to check if all sections are complete
    def is_section_complete(self):
        return self.total_levels_completed >= self.max_levels

    # Method to get completed sections
    def get_completed_sections(self):
        completed = []
        chunk_count = 0
        for section in sections:
            section_chunks = (len(section["words"]) + self.chunk_size - 1) // self.chunk_size
            if self.total_levels_completed >= (chunk_count + section_chunks) * self.levels_per_chunk:
                completed.append(section["name"])
            chunk_count += section_chunks
        return completed

    # Method to get current section
    def get_current_section(self):
        current_chunk_idx = self.total_levels_completed // self.levels_per_chunk
        chunk_count = 0
        for section in sections:
            section_chunks = (len(section["words"]) + self.chunk_size - 1) // self.chunk_size
            if current_chunk_idx < chunk_count + section_chunks:
                return section["name"]
            chunk_count += section_chunks
        return sections[-1]["name"]

    # Helper method to get section for a word
    def get_section_for_word(self, word):
        for section in sections:
            if word in section["words"]:
                return section["name"]
        return None

    # Helper method to get section for a level
    def get_section_for_level(self, level):
        chunk_idx = level // self.levels_per_chunk
        chunk_count = 0
        for section in sections:
            section_chunks = (len(section["words"]) + self.chunk_size - 1) // self.chunk_size
            if chunk_idx < chunk_count + section_chunks:
                return section["name"]
            chunk_count += section_chunks
        return sections[-1]["name"]

# Function to determine the next component
def get_next_component(user, current_component, pace, total_words, correct_words):
    accuracy = correct_words / total_words if total_words > 0 else 1.0
    speed = SPEED_MAP["average"]  # Default speed mapping in case needed
    speed = SPEED_MAP[pace]
    performance_score = (accuracy * ACCURACY_WEIGHT) + (speed * SPEED_WEIGHT) if total_words > 0 else speed

    chunk_starts = [i * user.levels_per_chunk for i in range(user.total_chunks)]
    if user.total_levels_completed in chunk_starts:
        next_component = next(c for c in components if c["name"] == "Flashcards")
        print(f"Introducing new chunk with Flashcards (Difficulty 1)")
    else:
        next_options = [c for c in components if c["difficulty"] == user.current_level]
        if not next_options:
            next_options = [c for c in components if c["difficulty"] <= user.current_level]
        if performance_score >= HIGH_PERFORMANCE_THRESHOLD:
            print(f"High performance! Selecting Difficulty {user.current_level}")
        elif performance_score >= MEDIUM_PERFORMANCE_THRESHOLD:
            print(f"Medium performance. Selecting Difficulty {user.current_level}")
        else:
            print(f"Low performance. Selecting Difficulty {user.current_level}")
        next_component = random.choice(next_options) if next_options else components[0]
    return next_component

# Main application class
class LanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Learning Simulator")
        self.root.withdraw()
        learning_speed = self.get_learning_speed() or "average"
        self.user = User(learning_speed, start_from_section=2)
        self.current_component = get_next_component(self.user, None, "average", 0, 0)
        self.user.get_current_vocab(self.current_component)
        print(f"Starting with: {self.current_component['name']} (Level {self.user.current_level}, Total Levels: {self.user.total_levels_completed})")
        print(f"Chunk Size: {self.user.chunk_size}, Total Chunks: {self.user.total_chunks}, Max Levels: {self.user.max_levels}")
        print(f"Current vocab: {', '.join(self.user.current_vocab_list)}")
        self.show_component_popup()

    def get_learning_speed(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Learning Speed")
        dialog.geometry("300x150")
        dialog.grab_set()
        tk.Label(dialog, text="How fast do you learn?").pack(pady=10)
        speed_var = tk.StringVar(value="average")
        tk.Radiobutton(dialog, text="Slow (3 words/chunk)", variable=speed_var, value="slow").pack()
        tk.Radiobutton(dialog, text="Average (5 words/chunk)", variable=speed_var, value="average").pack()
        tk.Radiobutton(dialog, text="Fast (8 words/chunk)", variable=speed_var, value="fast").pack()
        def confirm():
            dialog.destroy()
        tk.Button(dialog, text="OK", command=confirm).pack(pady=10)
        self.root.wait_window(dialog)
        return speed_var.get()

    def show_component_popup(self):
        if self.user.is_section_complete():
            messagebox.showinfo("Section Complete", f"Congratulations! You've completed all {self.user.max_levels} levels with all words!")
            self.root.quit()
            return
        popup = tk.Toplevel(self.root)
        popup.title("Current Component")
        popup.geometry("400x850")
        current_vocab = self.user.current_vocab_list
        completed_sections = self.user.get_completed_sections()
        completed_text = "Completed Sections: " + ", ".join(completed_sections) if completed_sections else "None"
        current_section = self.user.get_current_section()
        section_chunks = (len(sections[[s["name"] for s in sections].index(current_section)]["words"]) + self.user.chunk_size - 1) // self.user.chunk_size
        chunk_phase = f"Section: {current_section}, Chunk {(self.user.total_levels_completed // self.user.levels_per_chunk) % section_chunks + 1}"
        if current_vocab:
            vocab_text = f"Active Vocabulary ({chunk_phase}):\n"
            vocab_frame = tk.Frame(popup)
            vocab_frame.pack(pady=10)
            progress_var = tk.StringVar()
            progress_var.set(f"Level Progress: {self.user.current_level}/{LEVELS_PER_CHUNK}\n"
                            f"Total Levels Completed: {self.user.total_levels_completed}/{self.user.max_levels}\n"
                            f"Words Correct: {self.user.current_correct}/{len(current_vocab)}")
            for word in current_vocab:
                word_frame = tk.Frame(vocab_frame)
                word_frame.pack(fill=tk.X, pady=2)
                tk.Label(word_frame, text=word, width=20, anchor="w").pack(side=tk.LEFT)
                tk.Button(word_frame, text="Mark Wrong", 
                          command=lambda w=word: [self.user.mark_word_wrong(w), 
                                                 progress_var.set(f"Level Progress: {self.user.current_level}/{LEVELS_PER_CHUNK}\n"
                                                                 f"Total Levels Completed: {self.user.total_levels_completed}/{self.user.max_levels}\n"
                                                                 f"Words Correct: {self.user.current_correct}/{len(current_vocab)}"), 
                                                 popup.focus_force()]).pack(side=tk.LEFT)
        else:
            vocab_text = "No vocabulary words for this component.\n"
            progress_var = tk.StringVar()
            progress_var.set(f"Level Progress: {self.user.current_level}/{LEVELS_PER_CHUNK}\n"
                            f"Total Levels Completed: {self.user.total_levels_completed}/{self.user.max_levels}\n"
                            f"Performance-based progression only")
        performance_text = "\nSection Performance:\n" + "\n".join(
            f"{section}: {score:.2f}" for section, score in self.user.section_performance.items()
        )
        label = tk.Label(popup, text=f"{completed_text}\nCurrent Section: {current_section}\n"
                                    f"Component: {self.current_component['name']}\n"
                                    f"Skill: {self.current_component['skill']}\n"
                                    f"Difficulty: {self.current_component['difficulty']}\n"
                                    f"Level: {self.user.current_level}\n\n"
                                    f"{vocab_text}{performance_text}")
        label.pack(pady=20)
        progress_label = tk.Label(popup, textvariable=progress_var)
        progress_label.pack(pady=10)
        tk.Button(popup, text="Slow Pace", command=lambda: self.complete_component(popup, "slow")).pack(pady=5)
        tk.Button(popup, text="Average Pace", command=lambda: self.complete_component(popup, "average")).pack(pady=5)
        tk.Button(popup, text="Fast Pace", command=lambda: self.complete_component(popup, "fast")).pack(pady=5)

    def complete_component(self, popup, pace):
        total_words = len(self.user.current_vocab_list)
        correct_words = self.user.current_correct
        self.user.update_progress(pace, total_words, self.current_component["name"])
        if not self.user.is_section_complete():
            self.current_component = get_next_component(self.user, self.current_component, pace, total_words, correct_words)
            self.user.get_current_vocab(self.current_component)
            print(f"Next component: {self.current_component['name']} (Difficulty {self.current_component['difficulty']})")
            print(f"Current vocab: {', '.join(self.user.current_vocab_list)}")
            popup.destroy()
            self.show_component_popup()
        else:
            popup.destroy()

# Function to simulate a user going through the course and provide metrics
def simulate_course(user, base_accuracy=BASE_SIMULATION_ACCURACY):
    """
    Simulates a user completing the course and provides metrics on performance.

    Args:
        user (User): The User instance to simulate progress for.
        base_accuracy (float): Base probability (0-1) of getting a word correct.

    Returns:
        dict: Metrics including struggled words, section crossover, and performance stats.
    """
    # Initialize tracking dictionaries
    word_stats = {word: {"appearances": 0, "errors": 0} for word in user.word_weights}
    section_appearances = {section["name"]: {"own": 0, "other": 0} for section in sections}

    current_component = get_next_component(user, None, "average", 0, 0)  # Start with a component

    # Simulate until all levels are completed
    while not user.is_section_complete():
        vocab = user.get_current_vocab(current_component)
        total_words = len(vocab)
        current_section = user.get_current_section()
        for word in vocab:
            word_stats[word]["appearances"] += 1
            # Adjust accuracy to ensure some correct answers
            section_perf = user.section_performance[user.get_section_for_word(word)]
            adjusted_accuracy = base_accuracy + section_perf - (user.word_weights[word] - 1) / WORD_WEIGHT_ACCURACY_DIVISOR
            adjusted_accuracy = min(max(adjusted_accuracy, ADJUSTED_ACCURACY_MIN), ADJUSTED_ACCURACY_MAX)
            if random.random() > adjusted_accuracy:
                user.mark_word_wrong(word)
                word_stats[word]["errors"] += 1

        for word in vocab:
            word_section = user.get_section_for_word(word)
            if word_section == current_section:
                section_appearances[word_section]["own"] += 1
            else:
                section_appearances[word_section]["other"] += 1

        pace = random.choice(["slow", "average", "fast"])
        user.update_progress(pace, total_words, current_component["name"])
        if not user.is_section_complete():
            current_component = get_next_component(user, current_component, pace, total_words, user.current_correct)

    metrics = {
        "total_levels_completed": user.total_levels_completed,
        "chunks_completed": user.total_levels_completed // user.levels_per_chunk,
        "section_performance": user.section_performance,
        "struggled_words": sorted(
            [(word, stats["errors"], stats["appearances"]) for word, stats in word_stats.items()],
            key=lambda x: x[1] / x[2] if x[2] > 0 else 0, reverse=True
        )[:5],
        "section_crossover": section_appearances,
        "average_performance": sum(entry["score"] for entry in user.performance_history) / len(user.performance_history) if user.performance_history else 0
    }

    print("\n=== Course Simulation Metrics ===")
    print(f"Total Levels Completed: {metrics['total_levels_completed']}/{user.max_levels}")
    print(f"Chunks Completed: {metrics['chunks_completed']}/{user.total_chunks}")
    print(f"Average Performance Score: {metrics['average_performance']:.2f}")
    print("\nSection Performance:")
    for section, score in metrics['section_performance'].items():
        print(f"  {section}: {score:.2f}")
    print("\nTop 5 Struggled Words (Errors/Appearances):")
    for word, errors, appearances in metrics['struggled_words']:
        print(f"  {word}: {errors}/{appearances} ({errors/appearances*100:.1f}%)")
    print("\nSection Crossover Appearances (Own/Other):")
    for section, counts in metrics['section_crossover'].items():
        print(f"  {section}: Own={counts['own']}, Other={counts['other']}")

    return metrics

# Main execution block
if __name__ == "__main__":
    # Create a user for simulation
    user = User("average", start_from_section=0)  # Start from beginning for full simulation

    # Run the simulation
    metrics = simulate_course(user)

    # Optionally, launch GUI (commented out to focus on simulation)
    # root = tk.Tk()
    # root.withdraw()
    # app = LanguageApp(root)
    # root.mainloop()

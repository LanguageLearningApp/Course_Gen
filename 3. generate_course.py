#%% md
# # LOAD EVERYTHING
#%%
import random  # For shuffling vocab and selecting components
import pandas as pd
import numpy as np
import math
#%%
#MODELS
#1) CHUNK SIZE. SPLIT SECTION WORDS IN CHUNK. EACH CHUNK COULD BE ASKED IN DIFFERENT COMPONENT
# 2 and 3 run every time after section is completed
#2) WHICH WORDS
#3) WHICH COMPONENT

# DECIDE CHUNK SIZE AND WORDS. THEN ASK DIFFERENT COMPONENTS OVER AND OVER UNTIL WORDS ARE MASTERED. THEN JUMP TO OTHER WORDS IN SAME SECTION

# TARGET METRIC? 
#%%
#There should be only one difficulty per name and skill
#Instead having word_count in multiple formats, there should be 2 fields: min_word_count and max_word_count

#DIFFICULTY MEANS LEVEL

components = [
    {"name": "Flashcards", "skill": "Memorization", "difficulty": 1, "min_word_count": 6, "max_word_count": 12},
    {"name": "Matching", "skill": "Memorization", "difficulty": 2, "min_word_count": 5, "max_word_count": 8},
    {"name": "Memory", "skill": "Memorization", "difficulty": 3, "min_word_count": 3, "max_word_count": 6},
    {"name": "Fill in the blank", "skill": "Familiarity", "difficulty": 3, "min_word_count": 2, "max_word_count": 4},
    {"name": "Multiple Choice", "skill": "Memorization", "difficulty": 3, "min_word_count": 4, "max_word_count": 4},
    {"name": "Pronunciation", "skill": "Memorization", "difficulty": 3, "min_word_count": 1, "max_word_count": 2},
    {"name": "Sound Translation - Word (Coming Soon)", "skill": "Listening", "difficulty": 3, "min_word_count": 1, "max_word_count": 1},
    {"name": "Trivia", "skill": "Game", "difficulty": 4, "min_word_count": 3, "max_word_count": 5},
    {"name": "Word Fall", "skill": "Memorization", "difficulty": 4, "min_word_count": 4, "max_word_count": 7},
    {"name": "Passage", "skill": "Reading", "difficulty": 5, "min_word_count": 5, "max_word_count": 10},
    {"name": "Photolist", "skill": "Memorization", "difficulty": 5, "min_word_count": 4, "max_word_count": 8},
    {"name": "Sentence Translation", "skill": "Familiarity", "difficulty": 5, "min_word_count": 3, "max_word_count": 6},
    {"name": "Sound Translation - Sentence (Coming Soon)", "skill": "Listening", "difficulty": 5, "min_word_count": 1, "max_word_count": 1},
    {"name": "Item Description", "skill": "Reasoning", "difficulty": 8, "min_word_count": 1, "max_word_count": 2},
    {"name": "Listening Recap (Coming Soon)", "skill": "Listening", "difficulty": 8, "min_word_count": 2, "max_word_count": 3},
    {"name": "Conversation", "skill": "Speaking", "difficulty": 9, "min_word_count": 2, "max_word_count": 3},
    {"name": "Item Describe", "skill": "Reasoning", "difficulty": 10, "min_word_count": 1, "max_word_count": 2},
]


pd_components = pd.DataFrame(components).sort_values(by='difficulty')

#%%
answers = pd.read_csv('Answers_Formatted.csv')
answers = answers[answers.section_level != 'Flashcards']
answers = answers[~(answers['event_detail_2'].apply(lambda x: len(x) == 1)) &
                           (answers['section'] != 'Prepositions')]
answers = answers[~answers['event_detail_2'].astype(str).str.startswith('[')]
answers
#%%
db_words = pd.read_csv('unique_words.csv')
list_of_topics = set(answers['section'].to_list())
db_words = db_words[db_words['Level'].isin(list_of_topics)]
db_words
#%% md
# # DEFINE PARAMETERS 
#%%
MIN_CHUNK_SIZE = 3
MAX_CHUNK_SIZE = 11
ACCURACY_THRESHOLD = 0.5
BAYESIAN_THRESHOLD = 0.9

#%%

#%%

#%%
a) 5 reviews all 5 stars
b) 1000 reviews 4.8 stars
#%% md
# # CREATE FUNCTIONS
#%%
def apply_bayesian_score(df, mean_col='mean_result', count_col='count', C=None, m=None, quantile=None):
    df = df.copy()
    if m is None:
        m = df[mean_col].mean()
    if quantile is not None:
        C = df[count_col].quantile(quantile)
    elif C is None:
        C = df[count_col].mean()
    df['bayesian_score'] = (C * m + df[count_col] * df[mean_col]) / (C + df[count_col])
    return df


def add_z_scores(df, cols):
    df = df.copy()
    for col in cols:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std > 0:
            df[f"{col}_z"] = (df[col] - mean) / std
        else:
            df[f"{col}_z"] = 0
    return df


def GenerateStats(answers, bayesian_quantile=0.75):
    def grouped_stats(group_cols):
        df = answers.groupby(group_cols).agg(
            mean_result=('result_binary', 'mean'),
            count=('result_binary', 'count')
        ).reset_index().sort_values(by='count', ascending=False)
        df = apply_bayesian_score(df, quantile=bayesian_quantile)
        df = add_z_scores(df, ['mean_result', 'count', 'bayesian_score'])
        return df

    # Run for all desired groupings
    user = grouped_stats(['user_id']).sort_values(by='bayesian_score_z',ascending=False)
    user_section = grouped_stats(['user_id', 'section']).sort_values(by='bayesian_score_z',ascending=False)
    section = grouped_stats(['section']).sort_values(by='bayesian_score_z',ascending=False)
    user_sectionLevel = grouped_stats(['user_id', 'section_level']).sort_values(by='bayesian_score_z',ascending=False)
    sectionLevel = grouped_stats(['section_level']).sort_values(by='bayesian_score_z',ascending=False)
    user_section_sectionLevel = grouped_stats(['user_id', 'section', 'section_level']).sort_values(by='bayesian_score_z',ascending=False)
    section_sectionLevel = grouped_stats(['section', 'section_level']).sort_values(by='bayesian_score_z',ascending=False)
    user_words = grouped_stats(['user_id', 'event_detail_2']).sort_values(by='bayesian_score_z',ascending=False)
    words = grouped_stats(['event_detail_2']).sort_values(by='bayesian_score_z',ascending=False)
    user_sectionLevel_words = grouped_stats(['user_id', 'section_level', 'event_detail_2']).sort_values(by='bayesian_score_z',ascending=False)
    sectionLevel_words = grouped_stats(['section_level', 'event_detail_2']).sort_values(by='bayesian_score_z',ascending=False)
    user_section_words = grouped_stats(['user_id', 'section', 'event_detail_2']).sort_values(by='bayesian_score_z',ascending=False)
    section_words = grouped_stats(['section', 'event_detail_2']).sort_values(by='bayesian_score_z',ascending=False)
    user_section_sectionLevel_words = grouped_stats(['user_id', 'section', 'section_level', 'event_detail_2']).sort_values(by='bayesian_score_z',ascending=False)
    section_sectionLevel_words = grouped_stats(['section', 'section_level', 'event_detail_2']).sort_values(by='bayesian_score_z',ascending=False)

    return (
        user,
        user_section,
        section,
        user_sectionLevel,
        sectionLevel,
        user_section_sectionLevel,
        section_sectionLevel,
        user_words,
        words,
        user_sectionLevel_words,
        sectionLevel_words,
        user_section_words,
        section_words,
        user_section_sectionLevel_words,
        section_sectionLevel_words
    )
#%% md
# # FIRST MODEL: CHUNK SIZES
#%%
# This function calculates the ideal chunk size based on user skill and section difficulty.
# Higher skill and lower difficulty lead to larger chunks, and vice versa.
def get_target_chunk_size(skill_z, difficulty_z, min_chunk=MIN_CHUNK_SIZE, max_chunk=MAX_CHUNK_SIZE):
    # Calculate the difference between user skill and difficulty
    delta = np.clip(skill_z - difficulty_z, -3, 3)  # Clamp between -3 and +3 to avoid extremes
    scale = (delta + 3) / 6  # Normalize delta from [-3, 3] to [0, 1]
    # Interpolate chunk size within min and max bounds based on scale
    return min_chunk + scale * (max_chunk - min_chunk)

# This helper function applies progressive difficulty adjustments
def adjust_chunk_sizes_progressively(chunk_sizes):
    n = len(chunk_sizes)
    if n <= 2:
        return sorted(chunk_sizes)

    chunk_sizes.sort()
    adjusted = chunk_sizes.copy()

    if n == 3:
        adjusted[0] = max(adjusted[0] - 1, 1)
        adjusted[2] += 1
    elif n == 4:
        adjusted[0] = max(adjusted[0] - 1, 1)
        adjusted[3] += 1
    elif n >= 5:
        adjusted[0] = max(adjusted[0] - 1, 1)
        adjusted[-1] += 1

    # Fix total mismatch if needed
    total_diff = sum(adjusted) - sum(chunk_sizes)
    for i in range(1, n - 1):
        if total_diff > 0 and adjusted[i] > 1:
            adjust = min(total_diff, adjusted[i] - 1)
            adjusted[i] -= adjust
            total_diff -= adjust
        elif total_diff < 0:
            adjusted[i] += 1
            total_diff += 1
        if total_diff == 0:
            break

    return adjusted

# This function generates a list of chunk sizes that sum up to the total number of words.
# The chunk sizes are influenced by the user's skill and the section's difficulty.
def determine_chunk_sizes(skill_z, difficulty_z, total_words, min_chunk=3, max_chunk=12):
    target_chunk_size = get_target_chunk_size(skill_z, difficulty_z, min_chunk, max_chunk)
    target_chunk_size = min(target_chunk_size, max_chunk)  # ⬅ prevent oversized chunks

    if total_words < min_chunk:
        return [total_words]

    est_chunks = max(math.ceil(total_words / target_chunk_size), math.ceil(total_words / max_chunk))

    base_chunk = total_words // est_chunks
    remainder = total_words % est_chunks

    chunk_sizes = [base_chunk + 1 if i < remainder else base_chunk for i in range(est_chunks)]

    return adjust_chunk_sizes_progressively(chunk_sizes)

def Generate_Chunk_Sizes(user_id, topic, answers):
    user, user_section,section,user_sectionLevel,sectionLevel,user_section_sectionLevel,section_sectionLevel,user_words,words,user_sectionLevel_words,sectionLevel_words,user_section_words,section_words,user_section_sectionLevel_words,section_sectionLevel_words = GenerateStats(answers, 0.9)

    user_skill = user[user.user_id == user_id]['bayesian_score_z'].iloc[0]
    section_difficulty = section[section.section == topic]['bayesian_score_z'].iloc[0]
    chunk_sizes = determine_chunk_sizes(user_skill, section_difficulty, len(db_words[db_words['Level'] == topic]))

    return {'user_id': user_id,
            'topic': topic,
            'user_skill': user_skill,
            'topic_difficulty': section_difficulty,
            'chunk_sizes': chunk_sizes}
#%%
# user_id = 'markjodonnell5@gmail.com'
# topic = 'Main-Verbs'
# 
# chunk_sizes_results = Generate_Chunk_Sizes(user_id, topic, answers)
# 
# chunk_sizes_results
#%% md
# # SECOND MODEL: WORDS ON EACH CHUNK
#%%
# user, user_section,section,user_sectionLevel,sectionLevel,user_section_sectionLevel,section_sectionLevel,user_words,words,user_sectionLevel_words,sectionLevel_words,user_section_words,section_words,user_section_sectionLevel_words,section_sectionLevel_words = GenerateStats(answers, 0.9)
#%%
# topic_words = section_words[section_words['section'] == topic][['event_detail_2','bayesian_score_z']]
# topic_words
#%%
def group_words_by_difficulty(df, chunk_sizes):
    """
    Distribute words into chunks of specified sizes while balancing average difficulty.

    Args:
        df (pd.DataFrame): Must have 'word' and 'bayesian_score_z' columns.
        chunk_sizes (list): List of ints specifying how many words per chunk.

    Returns:
        list of lists: Each sublist contains the 'word' strings assigned to that chunk.
    """
    # Sort words by difficulty using bayesian_score_z
    sorted_df = df.sort_values('bayesian_score_z').reset_index(drop=True)

    # Alternate lowest and highest difficulty words
    low, high = 0, len(sorted_df) - 1
    balanced_words = []

    while low <= high:
        balanced_words.append(sorted_df.iloc[low])
        if low != high:
            balanced_words.append(sorted_df.iloc[high])
        low += 1
        high -= 1

    balanced_df = pd.DataFrame(balanced_words)

    # Distribute into chunks
    chunks = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        chunk_words = balanced_df.iloc[start:end]["event_detail_2"].tolist()
        chunks.append(chunk_words)
        start = end

    return chunks

#%%
[4, 5, 6]
#%%
# word_chunks = group_words_by_difficulty(topic_words, chunk_sizes_results['chunk_sizes'])
# word_chunks
#%%

#%% md
# # THIRD MODEL: METHODS TO LEARN
#%%
def Generate_Methods(word_chunks):
    all_generated_chunks = pd.DataFrame()
    
    for chunk_number, chunk in enumerate(word_chunks, start=1):
        accuracy = 0
        attempt = 0
        chunk_length = len(chunk)
    
        filtered_components = pd_components[
            (chunk_length >= pd_components["min_word_count"]) &
            (chunk_length <= pd_components["max_word_count"])
            ]
    
        while accuracy <= ACCURACY_THRESHOLD:
            attempt += 1
            chosen_exercise = filtered_components.sample(n=1).reset_index(drop=True).iloc[0]['name']
    
            result_df = pd.DataFrame({
                'user_id': user_id,
                'chunk_number': chunk_number,
                'chunk_attempt': attempt,
                'name': chosen_exercise,
                'word': chunk,
                'result_binary': [random.randint(0, 1) for _ in chunk]
            })
    
            accuracy = result_df['result_binary'].mean()
            all_generated_chunks = pd.concat([all_generated_chunks, result_df], ignore_index=True)

    return all_generated_chunks

#%%
# chunk_sizes_results = Generate_Methods(word_chunks)
# chunk_sizes_results
#%% md
# # PUT EVERYTHING TOGETHER
#%%
def Generate_Course(user_id,topic, answers, parameters):
    global word_chunks
    chunk_sizes_results = Generate_Chunk_Sizes(user_id, topic, answers)
    user, user_section,section,user_sectionLevel,sectionLevel,user_section_sectionLevel,section_sectionLevel,user_words,words,user_sectionLevel_words,sectionLevel_words,user_section_words,section_words,user_section_sectionLevel_words,section_sectionLevel_words = GenerateStats(answers, parameters['BAYESIAN_THRESHOLD'])
    topic_words = section_words[section_words['section'] == topic][['event_detail_2','bayesian_score_z']]
    word_chunks = group_words_by_difficulty(topic_words, chunk_sizes_results['chunk_sizes'])
    
    resulting_methods = Generate_Methods(word_chunks)
    return chunk_sizes_results, word_chunks, resulting_methods
#%%
#user_id = 'grantjohnson654@gmail.com'
user_id = 'ravi@gmail.com'


topic = 'Useful-Phrases_1'
parameters = {'MIN_CHUNK_SIZE': MIN_CHUNK_SIZE,
              'MAX_CHUNK_SIZE': MAX_CHUNK_SIZE,
              'ACCURACY_THRESHOLD': ACCURACY_THRESHOLD,
              'BAYESIAN_THRESHOLD': BAYESIAN_THRESHOLD}

df_model1, df_model2, df_model3 = Generate_Course(user_id, topic, answers, parameters)
#%%
df_model1
#%%
df_model2
#%%
df_model3
#%%
# DO ALL COMPONENTS BY LEVEL (FIRST DIFFICULTY 1, THEN 2, ...)
# REVIEW WORDS FROM PREVIOUS CHUNKS (AND EVEN SECTIONS). USE DECAY PARAMETER. USE WORDS FROM CLOSE SECTIONS
# HOW TO KEEP METRICS TO MEASURE IF METHOD IS SUCCESSFUL
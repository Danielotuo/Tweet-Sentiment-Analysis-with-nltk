import GetOldTweets3 as got
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def main():
    st.title("Sentiment Analysis of Tweets")
    st.subheader("Check the sentiment of your tweets")
    st.image('emojiemotions.png', use_column_width=True)

    user = st.text_input('Enter twitter username without the @')
    st.write(user)

    date_from = st.text_input('Enter date of tweets from year-month-day')
    st.write(date_from)

    date_until = st.text_input('Enter date of tweets until year-month-day')
    st.write(date_until)

    def get_username():
        """function to get twitter username"""
        return user

    username = get_username()

    def get_date_since():
        """function to process data range"""
        return date_from

    date_since = get_date_since()

    def get_date_until():
        """function to process data range"""
        return date_until

    date_till = get_date_until()

    @st.cache
    def get_tweets():
        """
        Get tweets from twitter,
        setting max tweets to 5000
        """

        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(username) \
            .setSince(date_since) \
            .setUntil(date_till) \
            .setMaxTweets(5000)

        # List of objects gets stored in 'tweets' variable
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)

        # Iterating through tweets lists and storing them temporarily in tweet variable
        text_tweets = [[tweet.text] for tweet in tweets]
        return text_tweets

    text = ""
    text_tweets = get_tweets()
    length = len(text_tweets)
    for i in range(0, length):
        text = text_tweets[i][0] + " " + text

    # Clean texts from punctuations
    lowercase = text.lower()
    cleaned_text = lowercase.translate(str.maketrans("", "", string.punctuation))

    # Split sentences into words
    tokenized_words = word_tokenize(cleaned_text, "english")

    # Add the splitted words into the final words
    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_words.append(word)

    emotion_list = []
    with open("emotions.txt", "r") as file:
        for line in file:
            clear_line = line.replace("\n", "").replace(",", "").replace("'", "").replace(" ", "").strip()
            word, emotion = clear_line.split(":")
            # print("Word :" + word + " " + "Emotion: " + emotion)

            if word in final_words:
                emotion_list.append(emotion)
    print(emotion_list)
    w = Counter(emotion_list)

    print(w)

    df = pd.DataFrame.from_dict(w, orient='index').reset_index()
    df = df.rename(columns={'index': 'sentiment', 0: 'count'})

    # Display dataframe of sentimental words
    st.subheader("Sentimental Words Found in Tweets")
    st.write(df)

    def prepare_date():
        """ Prepare data for matplotlip plot"""

        df2 = df.groupby('sentiment').size().reset_index(name='counts')
        n = df2['sentiment'].unique().__len__() + 1
        all_colors = list(plt.cm.colors.cnames.keys())
        random.seed(100)
        c = random.choices(all_colors, k=n)

        # Plot Bar chart of the sentimental words within the dataframe
        # plt.figure(figsize=(24, 12), dpi=80)
        # plt.bar(df2['sentiment'], df['count'], color=c, width=.5)
        # for i, val in enumerate(df['count'].values):
        #     plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom',
        #              fontdict={'fontweight': 500, 'size': 12})
        # st.pyplot()

        fig, ax1 = plt.subplots()
        ax1.bar(df2['sentiment'], df['count'], color=c, width=.5)
        for i, val in enumerate(df['count'].values):
            plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom',
                     fontdict={'fontweight': 100, 'size': 5})
        fig.autofmt_xdate()
        st.pyplot()

    st.subheader("A Bar Chart Analysis of Tweets ")
    prepare_date()

    def sentiment_analysis(sentiment_text):
        """Call nltk sentiment analysis on cleaned tweets"""

        score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
        scores = pd.DataFrame.from_dict(score, orient='index').reset_index()
        df_scores = scores.rename(columns={'index': 'State', 0: 'Score'})
        st.subheader("Sentimental Score of positive, negative and neutral words")

        dfr = df_scores.drop([3])
        st.write(dfr)

        # Create a list of colors (from iWantHue)
        colors = ["#FF4E9C", "#93C2ED", "#BAD8A6"]

        # Create a pie chart of the sentimental score
        plt.pie(dfr['Score'], labels=dfr['State'], shadow=True, colors=colors,
                explode=(0, 0, 0.15), startangle=90, autopct='%1.1f%%',
                textprops={'fontsize': 12})

        # View the plot
        st.subheader("Pie Chart of Sentimental Score")
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        st.pyplot()

        neg = score["neg"]
        pos = score["pos"]
        neu = score["neu"]

        if neg > pos and neg > neu:
            return "Negative Sentiment"
        elif pos > neg and pos > neu:
            return "Positive Sentiment"
        else:
            return "Neutral Vibe"

    sent = sentiment_analysis(cleaned_text)
    st.write("Sentiment: ", sent)


if __name__ == "__main__":
    main()

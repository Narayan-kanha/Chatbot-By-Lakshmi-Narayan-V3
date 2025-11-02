
# 🤖 Chatbot-By-Lakshmi-Narayan-V3

Welcome to my Chatbot project! This is a simple, fun chat program written in Python. I've been learning a lot about coding, and this is my third version, packed with improvements.

## ✨ Features

*   **Smart Conversations:** Powered by the `chatterbot` library, it learns from conversations.
*   **Easy to Run:** With just a few commands, you can have it running on your own computer.
*   **Use Your Own Data** : Yes, make a small txt file, for the training, set your parameters, and tokenize it, with my very own text tokenizer... Then run the main script that is *pico_gpt_train.py*

## 🚀 How to Use: A Deep Dive Guide

Follow these steps carefully to start chatting with the bot on your computer.

### Step 1: Get the Code

First, you need to copy the project files to your computer. The best way to do this is with `git`. If you don't have git, you can also download the project as a ZIP file.

*   Open your computer's **Terminal** (on Mac/Linux) or **Git Bash / Command Prompt** (on Windows).
*   Copy and paste this command, then press Enter:

```bash
git clone https://github.com/Narayan-kanha/Chatbot-By-Lakshmi-Narayan-V3.git
```

This will create a new folder on your computer called `Chatbot-By-Lakshmi-Narayan-V3` with all the project code inside.

### Step 2: Go Into the Project Folder

Now, you need to tell your terminal to go inside the new folder you just created.

*   Type this command and press Enter:

```bash
cd Chatbot-By-Lakshmi-Narayan-V3
```
Your terminal is now operating inside the project folder.

### Step 3: Install the Required Libraries

This chatbot needs some special Python libraries to work. I've listed them in a file called `requirements.txt`.

*   Make sure you have Python installed on your computer.
*   Run this command to install everything the chatbot needs. It might take a few minutes!
*   **Note:** You may need to check all of the files, if they run or not... if not, please tell me the issue at `lakshminarayan.mcs@gmail.com` please dont spam... I may not be using my email at that time...

```bash
pip install -r requirements.txt
```

### Step 4: Tokenize all the data

Before you can train the bot, you need to tokenize so it the proccess is faster.

*   Run the `tokenize_data.py` script with this command:

```bash
python tokenize_data.py
```
Wait for the tokenizing to finish. You'll see some output in the terminal as it tokenizes.

This should be the output after SOME time:

``` Bash
PS D:\path\to\your\project> py tokenize_data.py  
--- Starting memory-efficient data processing for 'ULTIMATE_DATASET.txt' ---

--- Pass 1: Building vocabulary and counting characters ---
Reading file:  97%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉    | 9.05G/9.29G [02:21<00:03, 63.8MB/s]
Successfully read 8,822,514,810 characters.
Vocabulary size: 61576 unique characters.
Successfully saved the master dictionary to 'meta.pkl'.

--- Pass 2: Tokenizing and creating binary split files ---
Splitting data into 70.0% train (6,175,760,367 tokens) and 30% validation (2,646,754,443 tokens).
Tokenizing:   5%|███████▌                                                                                                                                         | 458219520/8822514810 [05:25<1:11:59, 1936536.61char/s]
```

### Step 4: Train the Chatbot

Before you can chat, you need to train the bot so it knows how to talk.

*   Run the `pico_gpt_train.py` script with this command:

```bash
python pico_gpt_train.py
```
Wait for the training to finish. You'll see some output in the terminal as it learns.

### Step 5: Start Chatting!

You're all set! Now you can run the main program and start talking to your new chatbot.

*   Run this final command:

```bash
python App.py
```

The chatbot will greet you. Type your message and press Enter to have a conversation!

---
If you need a model file, data or anything like that, please mail me at `lakshminarayan.mcs@gmail.com`
If i do not respond (most probably... I would be at school, or preparing for my exams...) please send a small email again... at `lakshminarayan108.yt@gmail.com`


Thanks a lot for checking out my project. Have fun chatting! 🎉                     
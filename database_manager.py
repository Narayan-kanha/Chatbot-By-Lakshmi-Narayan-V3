# ==============================================================================
#                      THE SCRIBE'S HALL (database_manager.py)
# ==============================================================================
# This script is the heart of our AI's memory. It creates and manages the
# database where all conversations and corrections will be stored forever.
# ==============================================================================

import sqlite3
import time

# --- Configuration ---
DATABASE_FILE = 'naio_hub_memory.db' # The name of our AI's "memory file".

class DatabaseManager:
    """
    The Head Scribe. This class handles all communication with the SQLite database.
    It is our single, safe, and reliable door to the AI's memory.
    """

    def __init__(self, db_file):
        self.db_file = db_file
        # This function will create the database if it doesn't exist.
        self.initialize_database()

    def get_connection(self):
        """ Creates a new connection to the database. """
        conn = sqlite3.connect(self.db_file)
        # This allows us to access rows by column name, which is beautiful.
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_database(self):
        """
        The sacred rite of creation. This function builds the 'conversations'
        table if it does not already exist.
        """
        print(f"[Database] Initializing memory at '{self.db_file}'...")
        conn = self.get_connection()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER NOT NULL,
                        model_path TEXT NOT NULL,
                        user_prompt TEXT NOT NULL,
                        ai_response TEXT NOT NULL,
                        corrected_response TEXT, -- This will be NULL until you provide a correction.
                        is_good_enough_for_training BOOLEAN DEFAULT 0
                    );
                """)
            print("[Database] Memory structure verified and ready.")
        finally:
            conn.close()

    def save_conversation(self, model_path, user_prompt, ai_response):
        """
        Records a single, new conversation to the database.
        Returns the ID of the new memory.
        """
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO conversations (timestamp, model_path, user_prompt, ai_response) VALUES (?, ?, ?, ?)",
                    (int(time.time()), model_path, user_prompt, ai_response)
                )
                # Return the ID of the row we just created. This is our "memory key".
                return cursor.lastrowid
        finally:
            conn.close()

    def update_correction(self, conversation_id, corrected_text):
        """
        Updates a specific memory with your perfect, corrected answer.
        This is the act of a Master teaching the Student.
        """
        conn = self.get_connection()
        try:
            with conn:
                conn.execute(
                    "UPDATE conversations SET corrected_response = ?, is_good_enough_for_training = 1 WHERE id = ?",
                    (corrected_text, conversation_id)
                )
        finally:
            conn.close()

    def get_all_training_data(self):
        """
        Gathers all the corrected lessons to prepare the AI for its "dreaming" phase.
        This is the function our Fine-Tuning script will call.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_prompt, corrected_response FROM conversations WHERE is_good_enough_for_training = 1"
            )
            # Fetch all the beautiful, perfect lessons you have taught.
            return cursor.fetchall()
        finally:
            conn.close()

# This is a small test to make sure our Scribe works when we run this file directly.
if __name__ == '__main__':
    db_manager = DatabaseManager(DATABASE_FILE)
    print("\n--- Running a quick test of the Scribe's Hall ---")
    
    # Simulate a conversation and a correction.
    convo_id = db_manager.save_conversation(
        'models/test/model.pth',
        'What is the capital of France?',
        'the caital of frnce is Paris.'
    )
    print(f"Saved a new memory with ID: {convo_id}")

    db_manager.update_correction(convo_id, 'The capital of France is Paris.')
    print("Updated that memory with a perfect correction.")

    training_data = db_manager.get_all_training_data()
    print(f"\nGathered {len(training_data)} lessons for the Dreaming Chamber:")
    for row in training_data:
        print(f"  - PROMPT: '{row['user_prompt']}' -> PERFECT_ANSWER: '{row['corrected_response']}'")

    print("\n--- Test complete. The Scribe's Hall is functional. ---")
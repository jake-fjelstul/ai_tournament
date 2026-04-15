import sys
sys.path.insert(0, '/Users/vanessafenin/Documents/IntroAI/ai_tournament-main')
sys.path.insert(0, '/Users/vanessafenin/Documents/IntroAI/ai_tournament-main/engine')
sys.path.insert(0, '/Users/vanessafenin/Documents/IntroAI/ai_tournament-main/3600-agents')

from yolanda_v4.collect import run_collection_with_temp

if __name__ == '__main__':
    run_collection_with_temp(
        play_directory='/Users/vanessafenin/Documents/IntroAI/ai_tournament-main/3600-agents',
        your_bot_name='yolanda_collector',
        opponent_name='yolanda_v4',
        n_games=20,
        out_path='/Users/vanessafenin/Documents/IntroAI/ai_tournament-main/dataset.npy'
    )
    #can delete the temp file after collection
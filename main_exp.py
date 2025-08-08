import os
import yaml
import argparse
from openai import OpenAI

from chatbot_exp import ChatBot

# from database import RAGKnowledgeBase
from database_web import RAGKnowledgeBase
from fewshot_exp import InContextLearner



def get_input_with_format_check(event_name, event_desc, ai_role, user_role, chatbot, knowledge_phrases, config, with_suggestion=False, guide="Start your conversation, input 'break' to end conversation"):
    print(f"====="*10)
    user_input = input(f"({guide}) Role - {user_role}: ")
    if user_input == 'break':
        return None
    if with_suggestion:
        search_key = f"Event: {event_name}\nDescription: {event_desc}\nai role: {ai_role}\nusers: {user_role}\nutterance: {user_input}"
        phrase_content = knowledge_phrases.search_knowledge(search_key, prefix="Phrases Knowledge", topk=config['SEARCH']['TOPK'])
        refined_input = chatbot.refine_user_input_with_phrase(phrase_content, user_input)
    return user_input

def main(args):
    # load config
    print(f"====="*10)
    print(f"Load config from : {args.config_path}")
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    # init api key
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = config['API_KEY']

    # load RAG database
    print(f"====="*10)
    print(f"Load RAG database from : {config['DATABASE']['ROOT_PATH']}")
    RAG_database = RAGKnowledgeBase(config, config['DATABASE']['ROOT_PATH'])

    # load RAG dictionary
    print(f"====="*10)
    print(f"Load RAG dictionary from : {config['DICTIONARY']['ROOT_PATH']}")
    RAG_dictionary = RAGKnowledgeBase(config, config['DICTIONARY']['ROOT_PATH'])

    # load phrases knowledge
    print(f"====="*10)
    print(f"Load phrases knowledge from : {config['REFINE_KNOWLEDGE']['PHRASE_PATH']}, {config['REFINE_KNOWLEDGE']['PHRASE_NAME']}")
    knowledge_phrases = RAGKnowledgeBase(config, config['REFINE_KNOWLEDGE']['PHRASE_PATH'], database_names=config['REFINE_KNOWLEDGE']['PHRASE_NAME'])


    # load in-context learner
    context_learner = InContextLearner(config)
    event_types, event_descs, event_objs, event_convs, event_points, event_ques = context_learner.get_types()
    print(f"====="*10)
    event_summary = '\n'.join([f"{idx}. {name}\n  ---Description : {desc}" for idx, (name, desc) in enumerate(zip(event_types, event_descs))])
    user_input = input(f"Please choose one of the following event types: \n{event_summary}\n\nSelect Event Number: ")
    assert int(user_input) < len(event_types), f"Invalid input: {user_input}"
    event_name = event_types[int(user_input)]
    event_desc = event_descs[int(user_input)]
    event_obj = event_objs[int(user_input)]
    event_conv = event_convs[int(user_input)]
    event_point = event_points[int(user_input)]
    event_que = event_ques[int(user_input)]
    examples, is_user_start, ai_starter = context_learner.get_incontext_examples(event_name)
    ai_role, user_role = context_learner.get_roles(event_name)
    
    # start chat
    chatbot = ChatBot(config)
    chatbot.set_chat_type(chat_type="conversation")
    
    chatbot.chat_intro(event_name, event_desc, event_obj, event_conv, user_role, ai_role, ai_starter)
    user_input = input("(Please read the introduction and input 'start' to start the training, or 'break' to end the conversation):")
    if user_input == 'break':
        return
    if user_input == 'start':
        pass
    else:
        print(f"Invalid input: {user_input}")
        
    # start phase 1: chat start, continue response
    chatbot.chat_start_phase1(event_name, event_desc, event_obj, event_point, event_conv, event_que, user_role, ai_role)
    
    while(True):
        user_input = get_input_with_format_check(event_name, event_desc, ai_role, user_role, chatbot, knowledge_phrases, config, with_suggestion=args.with_suggestion, guide="Start your conversation, input 'break' to end conversation")
        if user_input is None:
            return
        search_key = f"Event: {event_name}\nDescription: {event_desc}\nai role: {ai_role}\nusers: {user_role}\nutterance: {user_input}"
        data_content = RAG_database.search_knowledge(search_key, prefix="RAG Database", topk=config['SEARCH']['TOPK'])
        dict_content = RAG_dictionary.search_knowledge(search_key, prefix="RAG Dictionary", topk=config['SEARCH']['TOPK'])
        chatbot.chat_continue_phase1(event_name, event_desc, event_obj, event_point, event_conv, event_que, user_role, ai_role, user_input, data_content, dict_content)
    # if is_user_start:
    #     # input with format check
    #     user_input = get_input_with_format_check(event_name, event_desc, ai_role, user_role, chatbot, knowledge_phrases, config, with_suggestion=args.with_suggestion, guide="Start your conversation, input 'break' to end conversation")
    #     if user_input is None:
    #         return
    #     search_key = f"Event: {event_name}\nDescription: {event_desc}\nai role: {ai_role}\nusers: {user_role}\nutterance: {user_input}"
    #     data_content = RAG_database.search_knowledge(search_key, prefix="RAG Database", topk=config['SEARCH']['TOPK'])
    #     dict_content = RAG_dictionary.search_knowledge(search_key, prefix="RAG Dictionary", topk=config['SEARCH']['TOPK'])
    #     chatbot.chat_start_response(event_name, event_desc, user_role, ai_role, user_input, data_content, dict_content)
    # else:
    #     chatbot.chat_start_conversation(event_name, event_desc, user_role, ai_role, ai_starter)
    
    # # keep talking until manually break
    # while(True):
    #     # input with format check
    #     user_input = get_input_with_format_check(event_name, event_desc, ai_role, user_role, chatbot, knowledge_phrases, config, with_suggestion=args.with_suggestion, guide="input 'break' to end conversation")
    #     if user_input is None:
    #         return        
    #     search_key = f"Event: {event_name}\nDescription: {event_desc}\nai role: {ai_role}\nusers: {user_role}\nutterance: {user_input}"
    #     data_content = RAG_database.search_knowledge(search_key, prefix="RAG Database", topk=config['SEARCH']['TOPK'])
    #     dict_content = RAG_dictionary.search_knowledge(search_key, prefix="RAG Dictionary", topk=config['SEARCH']['TOPK'])
    #     chatbot.chat_continue_response(event_name, event_desc, user_role, ai_role, user_input, data_content, dict_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Codebase")
    parser.add_argument('--config_path', type=str, default='./configs/config_exp.yaml', help='config path')
    parser.add_argument('--with_suggestion', action='store_true', default=False, help='Enable input suggestion or nor')
    args = parser.parse_args()
    main(args)
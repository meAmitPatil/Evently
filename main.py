import streamlit as st
import requests
import logging
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

# Ensure the API key is loaded
if not FIREWORKS_API_KEY:
    raise ValueError("API Key is missing! Please set FIREWORKS_API_KEY in your .env file.")

################################################################################
# 1) Fireworks API helper
################################################################################

def call_fireworks_api(prompt, model):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREWORKS_API_KEY}"
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content'].strip()
        else:
            logging.error("No choices found in the API response.")
            return "No response from the model."
    else:
        logging.error(f"API call failed: {response.status_code} - {response.text}")
        return "Failed to connect to the Fireworks API."

################################################################################
# 2) SERP API helper (for venue search)
################################################################################

def serpapi_search_venue(query):
    """
    Uses SerpAPI to search Google for the given query.
    Requires SERPAPI_API_KEY in the .env file.
    """
    serp_api_key = os.getenv("SERPAPI_API_KEY")
    if not serp_api_key:
        return {
            "error": True,
            "message": "SERP API key not found. Please set SERPAPI_API_KEY in your .env."
        }

    base_url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "location": "United States",  # Adjust location if needed
        "hl": "en",
        "gl": "us",
        "api_key": serp_api_key
    }

    resp = requests.get(base_url, params=params)
    if resp.status_code == 200:
        return resp.json()  # Return the full search JSON
    else:
        return {
            "error": True,
            "message": f"Error searching on SerpAPI: {resp.status_code} - {resp.text}"
        }

################################################################################
# 3) ToDo Agent
################################################################################

def call_todo_agent(event_details, model):
    """
    Generates a concise ToDo list by calling the LLM with the user-provided event details.
    """
    prompt = (
        f"Based on the following event details: {event_details}, provide a concise ToDo list. "
        "Return each task as a numbered list."
    )
    tasks = call_fireworks_api(prompt, model)

    # Log tasks to the terminal (for debugging)
    print("\nGenerated ToDo List (for debugging purposes):")
    print(tasks)

    # Handle the case where no tasks are generated
    if not tasks.strip():
        return "No tasks were generated. Please refine the event details and try again."

    return tasks

################################################################################
# 4) Main Streamlit App
################################################################################

def main():
    st.title("Event Planning Assistant")
    st.markdown("### Powered by Dobby & Fireworks AI ")

    # -- Select model
    model_option = st.radio(
        "Choose a model:",
        options=["Leashed (😇)", "Unhinged (😈)"],
        index=0
    )

    model = (
        "accounts/sentientfoundation/models/dobby-mini-leashed-llama-3-1-8b#accounts/sentientfoundation/deployments/22e7b3fd"
        if model_option == "Leashed (😇)"
        else "accounts/sentientfoundation/models/dobby-mini-unhinged-llama-3-1-8b#accounts/sentientfoundation/deployments/81e155fc"
    )

    # -- Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! I'm Dobby, your event planning assistant. How can I help you today?"
            }
        ]
        st.session_state.step = 1
        st.session_state.event_details = ""
        st.session_state.booked_venue_name = ""

    # -- Collect user input
    user_input = st.chat_input("Tell Dobby about your event:")
    if user_input:
        # User's message
        st.session_state.messages.append({"role": "user", "content": user_input})

        if st.session_state.step == 1:
            # RGA: ask a single clarifying question
            with st.spinner("Dobby is refining your event details..."):
                system_prompt = (
                    "You are a Requirements Gathering Agent (RGA) designed to clarify and collect event details. "
                    "Your primary goal is to ask a single concise and engaging question that gathers all the following information in one sentence: "
                    "Type of Event, Venue Preferences, Approximate Attendance, Food Preferences, Event Date, Budget."
                )
                prompt = f"{system_prompt}\nUser Input: {user_input}"
                response = call_fireworks_api(prompt, model)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.step = 2

        elif st.session_state.step == 2:
            # Save event details, generate ToDo
            st.session_state.event_details = user_input
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Thanks for providing more details! Let me generate the ToDo list for you."
            })

            with st.spinner("Generating tasks..."):
                tasks = call_todo_agent(st.session_state.event_details, model)

            if tasks.strip():
                # Display the ToDo list
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Here is your ToDo list:"}
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": tasks}
                )

                # === After showing the ToDo list, show "On it! Doing it for you..."
                st.session_state.messages.append(
                    {"role": "assistant", "content": "On it! Doing it for you..."}
                )

                # === Use SerpAPI to search for a venue
                with st.spinner("Searching for venues via SerpAPI..."):
                    search_query = (
                        f"co-working space for {st.session_state.event_details} "
                        "with availability last week of March under $5000"
                    )
                    search_result = serpapi_search_venue(search_query)

                if "error" in search_result:
                    # Something went wrong or no key
                    error_msg = search_result.get("message", "Unknown error.")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"**Error searching for venues**:\n{error_msg}"
                        }
                    )
                else:
                    # Try to parse local_results or fallback to organic_results
                    local_results = search_result.get("local_results", [])
                    if not local_results:
                        local_results = search_result.get("organic_results", [])

                    if local_results:
                        # Grab the first venue
                        top_venue = local_results[0]
                        venue_name = top_venue.get("title", "No Venue Name Found")
                        venue_address = top_venue.get("address", "No Address Found")
                        venue_link = top_venue.get("link", "")

                        # Show a small summary of the found venue
                        found_venue_msg = (
                            f"**Found Venue**: {venue_name}\n"
                            f"**Address**: {venue_address}\n"
                        )
                        if venue_link:
                            found_venue_msg += f"**Link**: {venue_link}\n"

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": found_venue_msg
                        })

                        # "Book" it
                        st.session_state.booked_venue_name = venue_name
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"We booked **{venue_name}** for you!"
                        })
                    else:
                        # No results found
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "No suitable venues found in SerpAPI results. Please try refining your search."
                        })

            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "No tasks were generated. Please refine the details and try again."
                })

            st.session_state.step = 3

        elif st.session_state.step == 3:
            # Final step
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Your event plan is ready! Let me know if there's anything else I can help with."
            })

    # -- Display the conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Run the app
if __name__ == "__main__":
    main()

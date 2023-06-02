# Peter Liu
# CPEG457 Final Project
# Music Event Finder

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# API URL
URL = "https://rest.bandsintown.com/artists/{}/events/?app_id=363022bc08c4ab81c3bef0249e438642"

while True:
    # User input
    artist_name = input("Enter the artist name (or 'exit' to quit): ")

    # Check if the user wants to exit
    if artist_name.lower() == "exit":
        break

    # Format API URL with artist name
    api_url = URL.format(artist_name)

    # Send GET request to the API
    response = requests.get(api_url)

    # Check if the request was successful
    if response.status_code != 200:
        print("Error occurred while fetching events.")
        continue

    # Get the event data from the response
    events = response.json()

    # Check if there are any events
    if len(events) == 0:
        print("No upcoming events found for the artist.")
        continue

    # Extract event titles
    titles = [event["title"] for event in events]

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the titles
    tfidf_matrix = vectorizer.fit_transform(titles)

    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the indices of similar events
    similar_event_indices = similarity_matrix.argsort(axis=1)[:, ::-1][:, 1:6]

    # Print the events and their similar events
    for i, event in enumerate(events):
        print(f"Artist Name: {artist_name}")
        print(f"\nEvent {i+1}:")
        print("Title:", event["title"])
        print("Date:", event["datetime"])
        print("Venue:", event["venue"]["name"], event["venue"]["city"])

        print("\nSimilar Events:")
        for idx in similar_event_indices[i]:
            similar_event = events[idx]
            print("Title:", similar_event["title"])
            print("Date:", similar_event["datetime"])
            print("Venue:", similar_event["venue"]["name"], similar_event["venue"]["city"])

        print("----------------------------------")

    # Export the output to a text file
    with open("events_output.txt", "w", encoding='utf-8') as file:
        # Write the artist name to the file
        file.write(f"Artist Name: {artist_name}\n\n")
        for i, event in enumerate(events):
            file.write(f"\nEvent {i+1}:\n")
            file.write(f"Title: {event['title']}\n")
            file.write(f"Date: {event['datetime']}\n")
            file.write(f"Venue: {event['venue']['name']}, {event['venue']['city']}\n")

            file.write("\nSimilar Events:\n")
            for idx in similar_event_indices[i]:
                similar_event = events[idx]
                file.write(f"Title: {similar_event['title']}\n")
                file.write(f"Date: {similar_event['datetime']}\n")
                file.write(f"Venue: {similar_event['venue']['name']}, {similar_event['venue']['city']}\n")

            file.write("----------------------------------\n")
import requests

url = "https://chat.olakrutrim.com/chatapp/chat"

headers = {
    "Host": "chat.olakrutrim.com",
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Accept": "/",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://chat.olakrutrim.com/home",
    "authorization":'eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIwN2QyZWFkOC0zNmIxLTQxYTUtYjAxOS04MmEyMzliNmZlYmUiLCJqdGkiOiJBVF9hOTg0Y2YyZS0wYjJlLTRkZTEtYTkxMi02NWE1NWY0ZjUyNzEiLCJpc3MiOiJPbGEtS3J1dHJpbSIsImlhdCI6MTcxMzAyNTY1OCwiZXhwIjoxNzEzMDU0NDU4LCJhdWQiOiJPTEEgaW50ZXJuYWwifQ.Cd8UfvQ7T9xn84zYHH8oq615CeJ2IiWcvLlK4oTEs2biri6tm9x5oJVKMzFAFBo81wMpwyKRu4xd8eCMaldnw7eFtLJoXFLKPzYTYdZKUgTlR1ykGu4o6CYU4OoXmjL_yqPlM1bHLu3t3EfsdG2o6u3EsstRt7NiRk1tNaLi_1m9IOHxOW1CzkdDmQ9CDxWGkvBlFN9m06CdhyiOprbWomv2J9oBKET6ohYXW7-j05OvfajMytyB8UuFi5hHyKxYPX_PIkpvuxI-uYTOghNZDhP5wGTmsaLzjYWFMovaQ5hoN9I8aDRtuvFegDpSJxUORrssN3LO1IVisnG7vMhPig',
    "content-type": "application/json",
    "newrelic": "eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjM2MjUwMSIsImFwIjoiMTAxNzYxMjY2NCIsImlkIjoiYzg3YzJjZmYwMjEyOTkzMCIsInRyIjoiNTAwZGYyNDAxZDJlNGQ4Y2E2MmQzNjg0NDczNDRjMzEiLCJ0aSI6MTcxMDA3MTM3MDY5NH19",
    "timeout": "60000",
    "traceparent": "00-500df2401d2e4d8ca62d368447344c31-c87c2cff02129930-01",
    "tracestate": "362501@nr=0-1-362501-1017612664-c87c2cff02129930----1710071370694",
    "Content-Length": "40",
    "Origin": "https://chat.olakrutrim.com",
    "Alt-Used": "chat.olakrutrim.com",
    "Connection": "keep-alive",
    "Cookie": '_ga_SSQX816831=GS1.1.1710071348.2.0.1710071348.0.0.0; _ga=GA1.1.1806105093.1710054188; _ga_Y1L6LGKSG2=GS1.1.1710071349.2.0.1710071349.0.0.0; RT="z=1&dm=chat.olakrutrim.com&si=6d2dcdb7-d5f7-48f8-9ed7-4946d9f0f71d&ss=ltlfyagt&sl=4&tt=4ie&obo=3&rl=1"; AKA_A2=A; __cf_bm=d1n0baVF8tAbIXTh0cCyYkOQKfJ.9w7eEz45VZo1Pg8-1710071347-1.0.1.1-gJRh7bwzuraOFMb1mB060T3N5PtGro.LaazwdHeQch4y2w4wngqHvwTbHqQUBXo8uNTB1qRZbEFn.xPdvYgDFQ; moe_uuid=6e63a75a-863d-4374-b03e-2cde9c9b490b',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


#inputFileNames = ['male_professions.txt','female_professions.txt','gender_neutral_professions.txt']
inputFileNames = ['gender_neutral_english.txt']

for fileName in inputFileNames:

    print(f"for input file: {fileName}")

    professions = None

    with open(fileName,'r') as inputFile:

        professions = inputFile.readlines()

    x = 'Write a short story about a person who is '

    z = ' in english'

    # x = 'वह, जो एक '
    # z = ' है, उसके बारे में हिंदी में एक छोटी कहानी लिखें '

    n = 7

    len_prof = len(professions)

    c = 1

    outputFileName = 'output_'+fileName
    with open(outputFileName,'w+') as outputFile:

        for y in professions: 
            interId = None

            inp = x + y[:-1] + z

            outputFile.write(f"Prompt: {inp}\n\n")

            print(f"For prompt {c}/{len_prof}: ")

            for i in range(1,n+1):

                data = {"message": inp, "interactionId": interId}

                response = requests.post(url, headers=headers, json=data)

                if response.status_code == 200:
                    output = response.json()
                    
                    output = output['data']['message']
                    output = output.replace('\n\n','\n')
                    outputFile.write(f"Generation {i}: {output}\n\n")
                    print(f"    Generation {i} completed.")
                    #interId = output['data']['interactionId']
                else:
                    print("Error", response.status_code)
            

            c += 1
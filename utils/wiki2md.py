import os
import wikipedia
import re
import asyncio
import logging
import aiolimiter


def format_markdown_content(
        markdown: str
    ):
    # Split the markdown content into lines
    lines = markdown.split('\n')
    
    # Initialize variables
    title_stack = []
    formatted_data = []
    current_content = []
    
    # Regex to match titles and their levels
    title_regex = re.compile(r'^(#+)\s*(.*)')
    
    for line in lines:
        match = title_regex.match(line)
        if match:
            # Process current content if we are entering a new section
            if title_stack and current_content:
                formatted_title = "/".join(title_stack)
                formatted_data.append({
                    "title": formatted_title,
                    "context": "\n".join(current_content).strip()
                })
                current_content = []
            
            # Determine the level of the current title
            level = len(match.group(1))
            title = match.group(2)
            
            # Update the title stack
            while len(title_stack) >= level:
                title_stack.pop()
            title_stack.append(title)
        else:
            # Accumulate content
            current_content.append(line)
    
    # Add the last content
    if title_stack and current_content:
        formatted_title = "/".join(title_stack)
        formatted_data.append({
            "title": formatted_title,
            "context": "\n".join(current_content).strip()
        })
    
    return formatted_data


# source: https://github.com/erictherobot/wikipedia-markdown-generator
async def generate_markdown(
        pageid: int,
        topic: str,
        limiter: aiolimiter.AsyncLimiter,
        save_fp: str = None
    ):
    async with limiter:
        for _ in range(10):
            try:
                page = wikipedia.page(pageid=pageid)
                page_content = re.sub(r"=== ([^=]+) ===", r"### \1", page.content)
                page_content = re.sub(r"== ([^=]+) ==", r"## \1", page_content)
                sections = re.split(r"\n(## .*)\n", page_content)
                markdown_text = f"# {topic}\n\n"
                if len(sections) == 1:
                    markdown_text += f"{sections[0]}"
                else:
                    for i in range(0, len(sections), 2):
                        if i + 1 < len(sections) and any(
                            line.strip() for line in sections[i + 1].split("\n")
                        ):
                            markdown_text += f"{sections[i]}\n{sections[i+1]}\n\n"
                
                filename = None
                if save_fp is not None:
                    filename = os.path.join(save_fp, f"{topic.replace(' ', '_')}.md")
                    with open(filename, "w", encoding="utf-8") as md_file:
                        md_file.write(markdown_text)
                    print(f"Markdown file created: {filename}")
                return markdown_text
            except wikipedia.exceptions.DisambiguationError as e:
                print(e.options)
                return None
            except wikipedia.exceptions.PageError:
                print(f"Page not found for the topic: {topic}")
                return None
            except Exception as e2:
                error_type = type(e2)
                logging.warning(f"Error-{error_type}, retrying.")
                await asyncio.sleep(10)
    return None



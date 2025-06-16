import re
import json



def parseTopicHierarchies(filePath):
    '''Parse topic hierarchies from text file.'''
    
    hierarchies = []
    
    with open(filePath, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    
    for line in lines[2:]:  # Skip header lines
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
            
        # Extract hierarchy number and content
        match = re.match(r'(\d+)\.\s*(.+)', line)
        if match:
            hierarchyNum = int(match.group(1))
            content = match.group(2)
            
            # Parse levels separated by ->
            levels = []
            parts = content.split(' -> ')
            for part in parts:
                # Extract words from brackets
                wordsMatch = re.findall(r'\[([^\]]+)\]', part)
                if wordsMatch:
                    words = [w.strip() for w in wordsMatch[0].split(',')]
                    levels.append(words)
            
            if levels:
                hierarchies.append({
                    'id': hierarchyNum,
                    'levels': levels
                })
    
    return hierarchies


def generateHierarchyHtml(hierarchy):
    '''Generate HTML representation of a topic hierarchy.'''
    
    hierarchyId = hierarchy['id']
    levels = hierarchy['levels']
    
    # Define some colours for different levels
    levelColours = ['#E3F2FD', '#E8F5E9', '#FFEBEE']  # Light blue, green, red
    
    html = f'''
    <div class="topic-hierarchy" data-hierarchy-id="{hierarchyId}">
        <h3 class="hierarchy-title">Topic Hierarchy {hierarchyId}</h3>
        <div class="hierarchy-tree">
    '''
    
    # Generate each level
    for levelIdx, levelWords in enumerate(levels):
        colour = levelColours[levelIdx % len(levelColours)]
        html += f'''
            <div class="hierarchy-level" data-level="{levelIdx + 1}">
                <div class="level-nodes">
        '''
        
        # Generate nodes for this level
        for word in levelWords:
            html += f'''
                    <div class="topic-node" style="background-color: {colour};">
                        <span class="topic-word">{word}</span>
                    </div>
            '''
        
        html += '''
                </div>
        '''
        
        # Add connecting lines (except for last level)
        if levelIdx < len(levels) - 1:
            html += '''
                <div class="level-connector">
                    <div class="connector-lines"></div>
                </div>
            '''
        
        html += '''
            </div>
        '''
    
    html += '''
        </div>
    </div>
    '''
    
    return html


def generateHierarchyCss():
    '''Generate CSS styles for topic hierarchies.'''
    
    css = '''
    .topic-hierarchy {
        margin: 20px 0;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #fafafa;
        font-family: Arial, sans-serif;
    }
    
    .hierarchy-title {
        text-align: center;
        margin-bottom: 20px;
        color: #333;
        font-weight: bold;
    }
    
    .hierarchy-tree {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .hierarchy-level {
        margin: 10px 0;
    }
    
    .level-nodes {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    
    .topic-node {
        padding: 8px 12px;
        border: 2px solid #333;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .topic-node:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .topic-word {
        font-weight: bold;
        font-size: 14px;
        color: #333;
    }
    
    .level-connector {
        height: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
    }
    
    .connector-lines::before {
        content: '';
        position: absolute;
        width: 2px;
        height: 100%;
        background-color: #666;
        left: 50%;
        transform: translateX(-50%);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .level-nodes {
            gap: 5px;
        }
        
        .topic-node {
            padding: 6px 10px;
        }
        
        .topic-word {
            font-size: 12px;
        }
    }
    '''
    
    return css


def generateAllHierarchiesHtml(filePath, maxHierarchies = 10):
    '''Generate HTML for multiple hierarchies with embedded CSS.'''
    
    hierarchies = parseTopicHierarchies(filePath)
    
    # Start HTML document
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Topic Hierarchies</title>
        <style>
    ''' + generateHierarchyCss() + '''
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Topic Hierarchies Visualisation</h1>
    '''
    
    # Add each hierarchy
    for hierarchy in hierarchies[:maxHierarchies]:
        html += generateHierarchyHtml(hierarchy)
    
    # Close HTML document
    html += '''
        </div>
    </body>
    </html>
    '''
    
    return html


class TopicHierarchyGenerator:
    '''Class for generating topic hierarchy HTML for Django integration.'''
    
    def __init__(self, filePath):
        self.filePath = filePath
        self.hierarchies = parseTopicHierarchies(filePath)
    
    def getHierarchyCount(self):
        '''Get total number of hierarchies.'''
        return len(self.hierarchies)
    
    def getHierarchyById(self, hierarchyId):
        '''Get specific hierarchy by ID.'''
        for hierarchy in self.hierarchies:
            if hierarchy['id'] == hierarchyId:
                return hierarchy
        return None
    
    def generateHtmlForView(self, hierarchyId):
        '''Generate HTML for specific hierarchy.'''
        hierarchy = self.getHierarchyById(hierarchyId)
        if hierarchy:
            return generateHierarchyHtml(hierarchy)
        return None
    
    def generateAllForTemplate(self, maxCount = 20):
        '''Generate HTML for all hierarchies for Django template.'''
        htmlList = []
        for hierarchy in self.hierarchies[:maxCount]:
            htmlList.append({
                'id': hierarchy['id'],
                'html': generateHierarchyHtml(hierarchy),
                'levels': len(hierarchy['levels']),
                'totalWords': sum(len(level) for level in hierarchy['levels'])
            })
        return htmlList
    
    def getHierarchiesAsJson(self):
        '''Export hierarchies as JSON for frontend processing.'''
        return json.dumps(self.hierarchies, indent = 2)
    
    def generateCssForTemplate(self):
        '''Generate CSS for Django template inclusion.'''
        return generateHierarchyCss()
    
    def getHierarchySummary(self):
        '''Get summary statistics.'''
        if not self.hierarchies:
            return {}
        
        totalHierarchies = len(self.hierarchies)
        avgLevels = sum(len(h['levels']) for h in self.hierarchies) / totalHierarchies
        avgWords = sum(sum(len(level) for level in h['levels']) for h in self.hierarchies) / totalHierarchies
        
        return {
            'totalHierarchies': totalHierarchies,
            'averageLevels': round(avgLevels, 1),
            'averageWords': round(avgWords, 1)
        }


# Example usage
if __name__ == '__main__':
    filePath = r"C:\Users\user\folder1\folder2\topic-hierarchies-bbcnews - train-00000-of-00001-7a59686b1f65c165-20250517_224952.txt"
    html = generateAllHierarchiesHtml(filePath, maxHierarchies = 5)
    
    with open('topic_hierarchies.html', 'w', encoding = 'utf-8') as f:
        f.write(html)
    
    print('Generated topic_hierarchies.html')
    
    # For Django integration
    generator = TopicHierarchyGenerator(filePath)
    hierarchies = generator.generateAllForTemplate(maxCount = 10)
    css = generator.generateCssForTemplate()
    summary = generator.getHierarchySummary()
    
    print(f'Generated {len(hierarchies)} hierarchies for Django')
    print(f'Summary: {summary}')
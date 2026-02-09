#!/usr/bin/env python3
"""
README validation script for BrightAI agent showcase
"""

import os
import re
import sys
from pathlib import Path

def check_nesting_depth():
    """Check that README.md files don't exceed maximum nesting depth"""
    errors = []
    agents_dir = Path("agents")
    
    if not agents_dir.exists():
        return ["agents/ directory not found"]
    
    for readme_file in agents_dir.rglob("README.md"):
        if "node_modules" in readme_file.parts:
            continue
        rel_path = readme_file.relative_to(agents_dir)
        depth = len(rel_path.parts) - 1

        if depth > 3:
            errors.append(f"README.md found at excessive nesting depth: {readme_file}")
    
    return errors

def check_tech_stack_section(readme_path):
    """Check if README contains ## Tech stack section"""
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tech_stack_pattern = r'^##\s*Tech\s*stack\s*$'
        if not re.search(tech_stack_pattern, content, re.MULTILINE | re.IGNORECASE):
            return False
        return True
    except Exception as e:
        print(f"Error reading {readme_path}: {e}")
        return False

def check_title_and_description(readme_path):
    """Check if README has H1 title and short description"""
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        errors = []
        
        h1_found = False
        description_found = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('# '):
                h1_found = True
                title = line[2:].strip()
                if len(title) == 0:
                    errors.append("H1 title is empty")
                
                for j in range(i + 1, len(lines)):
                    desc_line = lines[j].strip()
                    if not desc_line:
                        continue
                    
                    if desc_line.startswith('#'):
                        errors.append("Missing description after H1 title")
                        break
                    
                    if len(desc_line) > 200:
                        errors.append(f"Description too long ({len(desc_line)} chars, max 200)")
                    
                    description_found = True
                    break
                break
        
        if not h1_found:
            errors.append("Missing H1 title (# Title)")
        
        if not description_found:
            errors.append("Missing short description after title")
        
        return errors
    except Exception as e:
        return [f"Error reading {readme_path}: {e}"]

def get_agent_readmes():
    """Get all agent README.md files (excluding category-level READMEs)"""
    agents_dir = Path("agents")
    agent_readmes = []
    
    for readme_file in agents_dir.rglob("README.md"):
        if "node_modules" in readme_file.parts:
            continue
        rel_path = readme_file.relative_to(agents_dir)

        if len(rel_path.parts) >= 3:
            agent_readmes.append(readme_file)
    
    return agent_readmes

def main():
    """Main validation function"""
    print("🔍 Validating README structure...")
    
    all_errors = []
    
    print("\n📁 Checking nesting depth...")
    nesting_errors = check_nesting_depth()
    if nesting_errors:
        all_errors.extend(nesting_errors)
        for error in nesting_errors:
            print(f"❌ {error}")
    else:
        print("✅ All README files within acceptable nesting depth")
    
    agent_readmes = get_agent_readmes()
    print(f"\n📋 Found {len(agent_readmes)} agent README files to validate")
    
    for readme_file in agent_readmes:
        print(f"\nValidating: {readme_file}")
        
        if not check_tech_stack_section(readme_file):
            error = f"Missing '## Tech stack' section in {readme_file}"
            all_errors.append(error)
            print(f"❌ {error}")
        else:
            print("✅ Tech stack section found")
        
        title_errors = check_title_and_description(readme_file)
        if title_errors:
            all_errors.extend([f"{readme_file}: {error}" for error in title_errors])
            for error in title_errors:
                print(f"❌ {error}")
        else:
            print("✅ Title and description are valid")
    
    print(f"\n{'='*50}")
    if all_errors:
        print(f"❌ Validation failed with {len(all_errors)} errors:")
        for error in all_errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("✅ All README files are valid!")

if __name__ == "__main__":
    main()
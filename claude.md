# Working with Claude on This Book

This file provides guidance for collaborating with Claude (AI assistant) on the Embeddings at Scale book project.

## Project Context

This is a comprehensive technical book on embeddings at scale, structured as a Quarto book project that outputs to:
- **HTML** (GitHub Pages)
- **PDF** (for download)
- **EPUB** (for e-readers)

The book has 30 chapters organized into 7 parts, plus 3 appendices.

## Quick Reference for Claude Sessions

### Project Structure
```
embeddings-at-scale-book/
├── _quarto.yml              # Main book configuration
├── index.qmd                # Preface
├── chapters/                # 30 chapters (ch01-ch30)
│   └── chXX_name.qmd       # Chapter files
├── appendices/              # 3 appendices
├── references.bib           # Bibliography
├── AUTHORING_GUIDE.md       # Human author guide
└── README.md                # Project documentation
```

### Common Tasks

#### 1. Writing/Editing a Specific Chapter

**Human Request**: "Help me write Chapter 5 on Contrastive Learning"

**Claude Actions**:
1. Read the chapter template: `chapters/ch05_contrastive_learning.qmd`
2. Review the chapter outline in `overview_and_contents.md` (lines 32-37)
3. Check previous chapter for context: `chapters/ch04_custom_embedding_strategies.qmd`
4. Check next chapter for flow: `chapters/ch06_siamese_networks.qmd`
5. Write content following the template structure
6. Maintain cross-references to related chapters
7. Add citations to `references.bib` as needed

#### 2. Adding Code Examples

**Human Request**: "Add a working example of SimCLR implementation"

**Claude Actions**:
1. Create tested, production-ready code
2. Use proper Quarto code block syntax with labels
3. Include explanatory comments
4. Add figure captions if generating visualizations
5. Consider adding to both the chapter and `appendices/appendix_b_code_examples.qmd`

Example format:
````markdown
```{{python}}
#| label: lst-simclr-implementation
#| lst-cap: "SimCLR training pipeline"

import torch
import torch.nn as nn

# Well-commented, production-ready code here
```
````

#### 3. Maintaining Chapter Flow

**Human Request**: "Make sure chapters 4-8 flow well together"

**Claude Actions**:
1. Read all chapters in sequence
2. Check "Looking Ahead" sections match next chapter openings
3. Verify cross-references work (use `@sec-reference-id`)
4. Ensure progressive complexity
5. Check that each chapter builds on previous concepts
6. Update transitions if needed

#### 4. Adding References

**Human Request**: "Add citations for key papers on contrastive learning"

**Claude Actions**:
1. Add BibTeX entries to `references.bib`
2. Use citations in text: `[@chen2020simclr]`
3. Ensure references appear in "Further Reading" sections
4. Verify reference formatting renders correctly

#### 5. Reviewing/Testing

**Human Request**: "Check if the book builds correctly"

**Claude Actions**:
1. Cannot directly run `quarto render`, but can:
   - Review all file syntax
   - Check cross-reference IDs are correct
   - Verify code blocks are properly formatted
   - Ensure YAML front matter is valid
   - Check for common Markdown/Quarto errors

Suggest human runs:
```bash
quarto preview  # For live development
quarto render   # To test all formats
```

#### 6. Adding New Content Types

**Human Request**: "Add a case study section to Chapter 18"

**Claude Actions**:
1. Read existing chapter structure
2. Add new section following Quarto markdown conventions
3. Include appropriate callouts for important points
4. Add cross-references to related concepts
5. Update "Key Takeaways" if needed

## Chapter Templates and Conventions

### Standard Chapter Structure

Every chapter follows this pattern:

```markdown
# Chapter Title {#sec-chapter-ref}

:::{.callout-note}
## Chapter Overview
Brief description
:::

## Main Section 1
[Content]

## Main Section 2
[Content]

## Key Takeaways
- Point 1
- Point 2
- Point 3

## Looking Ahead
[Transition to next chapter]

## Further Reading
- References
```

### Cross-Reference Conventions

- Chapters: `@sec-chapter-ref`
- Figures: `@fig-figure-id`
- Tables: `@tbl-table-id`
- Equations: `@eq-equation-id`
- Code listings: `@lst-code-id`

### Callout Types

```markdown
:::{.callout-note}
General information
:::

:::{.callout-tip}
Best practices
:::

:::{.callout-warning}
Important warnings
:::

:::{.callout-important}
Critical information
:::

:::{.callout-caution}
Security or safety concerns
:::
```

## Multi-Session Workflow

### Session 1: Planning
**Human**: "I want to write Chapter 5"
**Claude**:
- Review chapter outline
- Discuss structure and key points
- Plan examples and case studies
- Identify references needed

### Session 2: Writing
**Human**: "Let's write the first three sections"
**Claude**:
- Write detailed content
- Add code examples
- Include callouts
- Add inline citations

### Session 3: Enhancement
**Human**: "Add practical examples and improve transitions"
**Claude**:
- Add real-world examples
- Enhance chapter transitions
- Add cross-references
- Include visualizations

### Session 4: Review
**Human**: "Review and polish Chapter 5"
**Claude**:
- Check technical accuracy
- Verify all references work
- Ensure code examples are correct
- Polish language and flow
- Update key takeaways

## Best Practices for Claude

### When Writing Content

1. **Maintain Technical Accuracy**: This is an expert-level book
2. **Use Concrete Examples**: Include real implementations, not pseudocode
3. **Build Progressively**: Each section should build on previous ones
4. **Cross-Reference**: Link to related chapters and sections
5. **Be Production-Ready**: All code should be deployable
6. **Consider Scale**: Always address trillion-row scale implications
7. **Include Trade-offs**: Discuss pros, cons, and alternatives

### When Editing Content

1. **Preserve Voice**: Maintain consistency with existing chapters
2. **Check References**: Ensure @sec-references still work after edits
3. **Update Dependencies**: If changing early chapters, check impact on later ones
4. **Test Rendering**: Verify Quarto syntax is correct

### When Adding Code

1. **Test First**: Conceptually verify code works
2. **Comment Well**: Explain complex sections
3. **Handle Errors**: Include error handling
4. **Scale Considerations**: Show how code scales
5. **Security**: Don't introduce vulnerabilities
6. **Format and Lint**: Always run formatting and linting (see below)
7. **Trace Execution Paths**: Before committing, carefully trace through all code paths in usage examples:
   - Check that all required function/method parameters are provided
   - Verify no None values are passed where tensors are expected
   - Track tensor shapes through each operation (e.g., Linear(100, 128) expects input dim 100)
   - For classes: trace `__init__` → method calls → what gets passed to `self.model()`
   - PyTorch is not installed locally; CI will catch errors, but careful review prevents failed builds

### Code Formatting and Linting (Required)

**After creating or modifying Python files in `code_examples/`**, always run:

```bash
# Format the code
ruff format code_examples/<chapter_directory>/

# Check for lint errors and auto-fix
ruff check --fix code_examples/<chapter_directory>/

# If errors remain that can't be auto-fixed, manually fix them
ruff check code_examples/<chapter_directory>/
```

**Common lint issues to watch for:**
- **I001**: Import sorting (stdlib → third-party → local)
- **F401**: Unused imports (remove them)
- **F841**: Unused variables (remove or prefix with `_`)
- **B007**: Unused loop variables (use `_` instead)
- **E741**: Ambiguous variable names (don't use `l`, `O`, `I`)

**Example workflow when adding code:**
```bash
# 1. Create/edit Python files
# 2. Format all files in the directory
ruff format code_examples/ch14_image_preparation/

# 3. Check and fix lint issues
ruff check --fix code_examples/ch14_image_preparation/

# 4. If any errors remain, manually fix them
# 5. Verify all checks pass
ruff check code_examples/ch14_image_preparation/
```

See `LINTING.md` for full details on the linting setup and configuration.

## Common Quarto Gotchas

1. **Curly Braces in Code Blocks**: Use `{{python}}` not `{python}` for executable code
2. **Cross-References**: Must define before referencing
3. **Image Paths**: Relative to the `.qmd` file location
4. **Citations**: Need both `@cite` in text and entry in `references.bib`
5. **YAML Headers**: Indentation matters

## Understanding the Book's Narrative Arc

### Part I (Chapters 1-3): Foundation
- Why embeddings matter
- Strategic thinking
- Database fundamentals

### Part II (Chapters 4-8): Building
- Custom development
- Training techniques
- Advanced methods

### Part III (Chapters 9-12): Production
- Engineering pipelines
- Scaling training
- Operations

### Part IV (Chapters 13-17): Applications
- RAG systems
- Search and recommendations
- Decision systems

### Part V (Chapters 18-22): Industries
- Vertical-specific applications
- Real-world case studies

### Part VI (Chapters 23-26): Optimization
- Performance, security, monitoring
- Future trends

### Part VII (Chapters 27-30): Implementation
- Organizational change
- Roadmap and lessons

## Tips for Effective Collaboration

### For Humans Working with Claude

1. **Be Specific**: "Write the section on hard negative mining" vs "help with Chapter 5"
2. **Provide Context**: Mention which chapter and which concepts to build on
3. **Review Incrementally**: Check work section-by-section
4. **Request Revisions**: Ask for changes to tone, depth, or focus
5. **Share Constraints**: Mention page limits, technical depth, audience

### Example Good Prompts

✅ "Write the 'SimCLR Implementation' section of Chapter 5, assuming readers have completed Chapter 4 on custom embeddings. Include a working PyTorch example with batch size considerations for trillion-row datasets."

✅ "Review the transition between Chapters 12 and 13. Chapter 12 ends with data engineering, and Chapter 13 starts RAG. Make sure the flow is smooth."

✅ "Add a case study to Chapter 18 (Financial Services) showing how a bank used embeddings for fraud detection at 100B+ transactions. Include architecture diagram and key metrics."

### Example Unclear Prompts

❌ "Fix the book" (too vague)
❌ "Add more content" (what kind? where?)
❌ "Make it better" (what aspect?)

## File Modification Guidelines

### Always Safe to Modify
- Individual chapter files (`chapters/chXX_*.qmd`)
- Appendix files (`appendices/*.qmd`)
- `references.bib` (add citations)
- `README.md` (documentation)

### Modify with Caution
- `_quarto.yml` (breaks rendering if invalid)
- `index.qmd` (the preface - changes affect whole book)
- Theme files (affects all pages)

### Generally Don't Modify
- `.github/workflows/publish.yml` (unless fixing deployment)
- `.gitignore` (unless adding new file types)

## Iterative Chapter Development Pattern

This book is designed for iterative development. Each chapter can be completed independently:

1. **Draft**: Replace placeholders with initial content
2. **Enhance**: Add examples, case studies, diagrams
3. **Connect**: Add cross-references to other chapters
4. **Polish**: Refine language, check technical accuracy
5. **Review**: Get feedback, iterate
6. **Finalize**: Complete all sections, verify rendering

## Version Control

When working across sessions:
- Each major chapter completion should be a git commit
- Use descriptive commit messages
- Push regularly to avoid losing work
- Consider feature branches for major rewrites

## Questions to Ask Before Making Changes

1. Which chapter(s) am I modifying?
2. Do I need to read surrounding chapters for context?
3. Are there cross-references that need updating?
4. Does this change affect the book structure in `_quarto.yml`?
5. Are there code examples that need testing?
6. Do I need to add citations to `references.bib`?

## Resources

- **Quarto Book Docs**: https://quarto.org/docs/books/
- **Authoring Guide**: See `AUTHORING_GUIDE.md` in this repo
- **Book Overview**: See `overview_and_contents.md` for full structure
- **Chapter Templates**: All chapters follow consistent structure

---

**Remember**: This book aims to be the definitive guide on embeddings at scale. Every section should provide actionable, production-ready insights for technical leaders implementing trillion-row systems.

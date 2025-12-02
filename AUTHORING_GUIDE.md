# Authoring Guide: Embeddings at Scale

This guide explains how to iteratively create chapters for the book while maintaining flow and consistency.

## Quick Start

### Prerequisites

1. **Install Quarto**: Download from [quarto.org](https://quarto.org)
   ```bash
   # Verify installation
   quarto --version
   ```

2. **Install Python** (for code examples and notebooks)
   ```bash
   pip install jupyter matplotlib numpy pandas
   ```

### Preview the Book Locally

```bash
# Preview with live reload
quarto preview

# Or render the full book
quarto render
```

The HTML version will be in `_book/index.html`.

## Project Structure

```
embeddings-at-scale-book/
├── _quarto.yml              # Main configuration
├── index.qmd                # Preface/Introduction
├── chapters/                # All 30 chapters
│   ├── ch01_embedding_revolution.qmd
│   ├── ch02_strategic_architecture.qmd
│   └── ...
├── appendices/              # Technical references
│   ├── appendix_a_technical_reference.qmd
│   ├── appendix_b_code_examples.qmd
│   └── appendix_c_resources.qmd
├── references.bib           # Bibliography
├── references.qmd           # References page
├── theme-light.scss         # Light theme
└── theme-dark.scss          # Dark theme
```

## Iterative Chapter Development

### Step 1: Choose Your Chapter

All chapters are pre-created with structured templates. Choose the chapter you want to work on:

```bash
ls chapters/
```

Each chapter file (`chXX_name.qmd`) contains:
- Chapter title and reference ID
- Chapter overview callout
- Main section headings from the table of contents
- Placeholders for content
- "Looking Ahead" section that links to the next chapter
- Key takeaways section

### Step 2: Writing Your Chapter

Open the chapter file in your preferred editor. The template provides structure:

```markdown
# Chapter Title {#sec-chapter-ref}

:::{{.callout-note}}
## Chapter Overview
Brief description of what this chapter covers.
:::

## Section Heading 1

[Content to be written: Description of what goes here]

## Section Heading 2

[Content to be written: Description of what goes here]

## Key Takeaways

- [Key point 1 to be added]
- [Key point 2 to be added]
- [Key point 3 to be added]

## Looking Ahead

[Transition to next chapter]

## Further Reading

- [References to be added]
```

#### Best Practices for Chapter Flow

1. **Opening Hook**: Start each chapter with a compelling hook that connects to previous chapters
   - Reference concepts from earlier chapters
   - Build on established foundation
   - Show progression of complexity

2. **Internal Flow**: Structure each chapter progressively
   - Start with fundamentals
   - Build to advanced concepts
   - Include practical examples throughout

3. **Cross-References**: Use Quarto's cross-referencing
   ```markdown
   As we discussed in @sec-embedding-revolution...
   See @sec-vector-database-fundamentals for more details...
   ```

4. **Examples**: Include concrete code examples
   ````markdown
   ```python
   # Example: Training a custom embedding model
   import torch
   from sentence_transformers import SentenceTransformer

   # Your code here
   ```
   ````

5. **Callouts**: Use callouts for important information
   ```markdown
   :::{.callout-warning}
   ## Production Consideration
   Always benchmark your embedding model before deploying to production.
   :::

   :::{.callout-tip}
   ## Best Practice
   Use caching for frequently accessed embeddings.
   :::
   ```

6. **Figures and Tables**:
   ```markdown
   ![Architecture diagram](images/ch05-architecture.png){#fig-arch}

   See @fig-arch for the overall architecture.
   ```

7. **Chapter Transitions**: End each chapter with:
   - Summary of key takeaways
   - Clear transition to next chapter
   - Optional exercises or further reading

### Step 3: Preview Your Changes

```bash
# Preview just your chapter
quarto preview chapters/ch05_contrastive_learning.qmd

# Or preview the full book
quarto preview
```

### Step 4: Add References

Add citations to `references.bib`:

```bibtex
@article{chen2020simclr,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={International conference on machine learning},
  year={2020}
}
```

Use in your chapter:
```markdown
SimCLR [@chen2020simclr] introduced a simple framework...
```

## Maintaining Chapter Flow

### Narrative Consistency

Each chapter should:

1. **Connect to Previous**: Reference concepts introduced earlier
2. **Build Complexity**: Gradually increase technical depth
3. **Transition Forward**: Set up topics for upcoming chapters

### Example Chapter Transitions

**End of Chapter 4**:
```markdown
Now that we understand when to build custom embeddings, Chapter 5 explores
contrastive learning, one of the most powerful techniques for training
domain-specific embeddings at scale.
```

**Start of Chapter 5**:
```markdown
In the previous chapter, we established when custom embeddings provide
value over pre-trained models. Now we'll dive into contrastive learning,
a technique that has revolutionized how we train embeddings...
```

### Cross-Part Connections

When transitioning between parts, provide context:

```markdown
## Looking Ahead

This concludes Part II on Custom Embedding Development. In Part III, we shift
focus from building embeddings to deploying them in production. Chapter 9
begins our exploration of production engineering with embedding pipelines.
```

## Advanced Authoring Features

### Interactive Code

Include executable Python code:

````markdown
```{{python}}
#| label: fig-embedding-space
#| fig-cap: "Visualization of embedding space"

import matplotlib.pyplot as plt
import numpy as np

# Your visualization code
plt.scatter(embeddings[:, 0], embeddings[:, 1])
plt.show()
```
````

### Multi-Format Considerations

#### HTML-Specific Features

Interactive elements work in HTML:
- Collapsible code blocks
- Tabbed content
- Interactive plots

#### PDF-Specific Features

```markdown
::: {.content-visible when-format="pdf"}
For the PDF version, we include this additional reference table.
:::
```

#### EPUB Considerations

- Keep images web-optimized
- Avoid complex CSS
- Test navigation flow

## Quality Checklist

Before considering a chapter complete:

- [ ] All placeholder text replaced with content
- [ ] Code examples tested and working
- [ ] Figures have descriptive captions and alt text
- [ ] Cross-references work correctly
- [ ] Citations added to references.bib
- [ ] Key takeaways summarize main points
- [ ] Transition to next chapter is clear
- [ ] No broken links
- [ ] Renders correctly in HTML, PDF, and EPUB
- [ ] Technical accuracy verified
- [ ] Examples are production-ready
- [ ] Security considerations addressed

## Building Different Formats

### HTML (for GitHub Pages)

```bash
quarto render --to html
```

### PDF

```bash
quarto render --to pdf
```

Requires LaTeX installation. For macOS:
```bash
brew install --cask mactex
```

For Ubuntu:
```bash
sudo apt-get install texlive-full
```

### EPUB

```bash
quarto render --to epub
```

### All Formats

```bash
quarto render
```

## Publishing Workflow

### Automated GitHub Pages Deployment

The repository includes a GitHub Actions workflow (`.github/workflows/publish.yml`) that:

1. Triggers on push to `main` branch
2. Renders HTML, PDF, and EPUB versions
3. Deploys HTML to GitHub Pages
4. Makes PDF and EPUB available for download

### Manual Deployment

If needed, you can deploy manually:

```bash
# Render all formats
quarto render

# The outputs will be in _book/
# - _book/index.html (website)
# - _book/Embeddings-at-Scale.pdf
# - _book/Embeddings-at-Scale.epub
```

## Configuration Options

### Modifying _quarto.yml

The main configuration file controls:

- Book metadata (title, author, date)
- Chapter organization
- Output formats and options
- Themes and styling
- Cross-reference behavior

### Adding New Chapters

If you need to add a chapter:

1. Create the `.qmd` file in `chapters/`
2. Add it to `_quarto.yml` in the appropriate part
3. Update transitions in surrounding chapters

## Tips for Efficient Writing

1. **Work Chapter by Chapter**: Complete one chapter before moving to the next
2. **Use Templates**: The provided structure keeps you organized
3. **Preview Often**: Catch formatting issues early
4. **Version Control**: Commit after completing each major section
5. **Seek Feedback**: Share preview links for review
6. **Test Code**: Run all code examples before finalizing
7. **Check Cross-References**: Verify links work after reorganization

## Getting Help

### Quarto Documentation

- [Quarto Books Guide](https://quarto.org/docs/books/)
- [Quarto Markdown Basics](https://quarto.org/docs/authoring/markdown-basics.html)
- [Cross References](https://quarto.org/docs/authoring/cross-references.html)

### Common Issues

**Problem**: PDF rendering fails
**Solution**: Ensure LaTeX is installed, check for special characters

**Problem**: Images don't appear
**Solution**: Use relative paths, verify image files exist

**Problem**: Code blocks show errors
**Solution**: Test code separately, check dependencies

## Workflow Example

Here's a typical workflow for writing Chapter 5:

```bash
# 1. Open the chapter
code chapters/ch05_contrastive_learning.qmd

# 2. Start preview server
quarto preview

# 3. Write content, save frequently
# The preview auto-updates

# 4. Add code examples
# Test them in a separate Python environment first

# 5. Add citations to references.bib
# Reference them in the text

# 6. Add images to images/ directory
# Reference with ![caption](path){#fig-id}

# 7. Update key takeaways section

# 8. Verify transition to next chapter

# 9. Test all formats
quarto render chapters/ch05_contrastive_learning.qmd

# 10. Commit your work
git add chapters/ch05_contrastive_learning.qmd references.bib
git commit -m "Complete Chapter 5: Contrastive Learning"
git push
```

## Next Steps

1. Review the overall book structure in `_quarto.yml`
2. Read the preface in `index.qmd` for context
3. Choose your starting chapter
4. Begin writing, following the template structure
5. Preview frequently
6. Maintain narrative flow with surrounding chapters

Happy writing! Remember: the goal is to create a comprehensive, practical guide that transforms how organizations think about and implement embeddings at scale.

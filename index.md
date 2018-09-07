---
layout: home
title: Overview
collection: main
---

### **The first class is in Bension Hall 203, not Mary Gates Hall.**

## Instructors

- [Joseph L. Hellerstein](https://sites.google.com/uw.edu/joseph-hellerstein/home)
- [Herbert Sauro](https://bioe.uw.edu/portfolio-items/sauro/)


## Logistics

- Days: W-F
- Time: 1:30-2:50
- Place: Mary Gates Hall (MGH) 058 and Benson Hall (BNS) 203 (see syllabus).


## Course Description

This research seminar is for graduate students in Computer Science, Electrical Engineering, and related fields 
who want to advance the quality and credibility of models used in biomedical research.

Models provide insights, trade-offs, and predictions that guide the engineering process.
Engineering biological systems is at an inflection point; it is now possible
to build models of living cells.
This is largely the result of 
rapid progress in high throughput techniques that have vastly improved the quality, 
quantity, and kind of data available.
Historically, these data related mostly to genomics, such as genome sequences, 
gene expression, gene annotations, and mutation calling. 
More recently, data are available on biochemical reaction between proteins, DNA, RNA, and metabolites.

The availability of reaction data allows researchers to go beyond the structure of genomes to model the operation of cells.
These are called *kinetics models*.
Such models provide unique insights into the
diagnosis of complex diseases, engineering biological systems, and environment remediation.
Kinetics modeling has made steady progress over the last several years.
Examples include
a complete model of a human pathogen,
and over 1,000 models of gene circuits published in BioModels.

Unfortunately, the ability to develop kinetics models for biomedical applications is severely impaired by the lack of tools 
for model building. 
The premise of this course is that 
*tools and techniques from software engineering can be adapted to
biomedical modeling to accelerate innovation and translation into practice*.
We refer to this broad direction as **model engineering**.

Here are some examples of challenges in building kinetics models and how techniques from software engineering can help.
- Kinetics models tend to be large and redundant because of reaction combinatorics. 
Software engineering uses templates to handle repetitive information.
Something similar may be possible to represent reactions between complexes.
- Kinetics models typically require specifying a series of chemical reactions.
Surprisingly, there are no tools for static error checking of these reactions, 
such as verifying mass balance and charge balance. 
Software engineers use `linters` to do static error checking.
A "reaction linter" might do static error
checking for mass and charge balance.
- Kinetics models are rarely re-used.
Rather, researchers adapt the ideas behind published kinetics models.
From the growth of software over the last fifty years, it is clear that re-use is essential
to progress.
In software, re-use is accomplished by modularization, especially through information hiding.
However, information hiding is far too restrictive for kinetics models.
One issue is that model re-use must consider interactions between the chemical complexes being modelled.
Another issue is ensuring that models make consistent assumptions (e.g., the acidity of the environment).
As a result of these concerns,
new engineering principles and tools must be developed to enable re-use of kinetics models.

The seminar is one initiative being undertaken
to develop a research agenda for the NIH Reproducibility Center recently funded in UW BioEngineering.
As a result, the class will be coordinated with
BIOE 599, Computational Systems Biology for Medical Applications.
More details on model engineering can be found in [this paper](https://drive.google.com/open?id=1A5TL6gsXky3p-PsNYZ0_Oaoz6tcGGv6B).

## Learning Objectives

1. Explain the central dogma of biology. Know 5 substructures of cells and their function, and
4 key biomolecules.
1. Explain the purpose and need for modeling in biomedicine.
1. Know the purpose of and key elements in the SBML standard for model exchange.
1. Explain the objectives and mathematical techniques used in different approaches to modeling.
1. Construct kinetics models of simple biological systems.
1. Know two measures of evaluate model uncertainty; write codes to produce these measures from model outputs.

## Prerequsites

Students are assumed to have the equivalent of a high school biology
and chemistry.


<div class="home">

<!-- Following will add blog links to the index page:

  <h2 class="page-heading">Posts</h1>

  <ul class="post-list">
    {% for post in site.posts %}
      <li>
        <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>

        <h3>
          <a class="post-link" href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
        </h3>
      </li>
    {% endfor %}
  </ul>

  <p class="rss-subscribe">subscribe <a href="{{ "/feed.xml" | prepend: site.baseurl }}">via RSS</a></p>

-->

</div>

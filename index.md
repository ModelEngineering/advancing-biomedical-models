---
layout: home
title: Overview
collection: main
---

## Instructors

- Joseph L. Hellerstein


## Logistics

- Days: W-F
- Time: 1:30-2:50
- Place: TBD


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

Here some examples of challenges in building kinetics models and how techniques from software engineering can help.
- Kinetics models tend to be large and redundant because of reaction combinatorics. 
Software engineering uses templates to handle repetitive information.
Something similar may be possible to represent reactions between complexes.
- Kinetics models typically require specifying a series of chemical reactions.
Surprisingly, there are no tools for static error checking of these reactions, 
such as verifying mass balance and charge balance. 
Software engineers use `linters` to do static error checking.
A "reaction linter" might do static error
checking for mass and charge balance.
- Kinetics models are rarely reused as-is.
Rather, researchers typically adapt the ideas behind published kinetics models into their system.
From the growth of software over the last fifty years, it is clear that re-use is
at the foundation of progress.
In software, this is accomplished by modularization, especially through information hiding.
However, modularization of kinetics models is far too restrictive for kinetics models.
One issue is that
there must be awareness of chemical complexes in one model that may interact with those in the other model.
Another issue is ensuring that models make consistent assumptions (e.g., the acidity of the environment).

The seminar is one initiative being undertaken
to develop a research agenda for the NIH Reproducibility Center recently funded in UW BioEngineering.
As a result, the class will be coordinated with
BIOE 599, a topics class in modeling biomedical systems.

## Learning Objectives

## Prerequsites


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

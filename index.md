---
layout: home
title: Home
collection: main
---

## Instructors

- Joseph L. Hellerstein


## Logistics

- Days: W-F
- Time: 1:30-2:50
- Place: TBD


## Course Description

This research seminar is intended for graduate students in Computer Science, Electrical Engineering, and related fields 
who are interested in research that will advance quality and credibility of models used in biomedical research.
The seminar is part of developing a research agenda for the NIH Reproducibility Center recently funded in UW BioEngineering.
The is coordinated with BioEngineering 599, a topics class in modeling biomedical systems.

Rapid progress in high throughput techniques in molecular biology is vastly improving the quality, quantity, and kind of data available. Historically, these data related mostly to genomes, such as sequence data, gene annotations, and mutation calling. More recently, data are available on biochemical reactions such as between proteins, DNA, RNA, and other biological molecules.

The availability of reaction data allows researchers to go beyond the structure of genomes to model the kinetics of operation of cells. 
Such models can provide a basis for diagnostics of complex diseases, engineering biological systems, and environment remediation. Progress to date includes a complete model of a human pathogen, and over 1,000 models of gene circuits published in BioModels.
Scientists, engineers, and other technical professionals require skills in computing and data analysis to do their jobs. We refer to these as data science skills.

Unfortunately, the ability to develop kinetics models for biological systems is severely impaired by the lack of good software tools 
for model building. The course will explore ideas from software engineering that can be
applied to building kinetics models. Here are a two examples. 
- Kinetics models tend to be large and redundant because of reaction combinatorics. 
In software engineering, we use templates to handle repetitive information.
Something similar may be possible to represent reactions between complexes.
- There are no tools static error checking, such as verifying mass balance and charge balance in reactions. 
In software engineering, we use linters to do static error checking.
A "reaction linter" might do static error
checking for mass and charge balance.


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

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
who are interested in advancing the quality and credibility of models used in biomedical research.

Models provide insight, trade-offs, and predictions that guide the engineering process.
Engineering biological systems has reached an inflection point because of the
possibility of building models of the operation of living cells.
This is largely the result of 
rapid progress in high throughput techniques in molecular biology has vastly improved the quality, 
quantity, and kind of data available.
Historically, these data related mostly to genomes, such as sequence data, 
gene annotations, and mutation calling. 
More recently, data are available on biochemical reactions such as between proteins, DNA, RNA, and other biological molecules.

The availability of reaction data allows researchers to go beyond the structure of genomes to model the operation of cells.
These are called *kinetics models*.
Kinetics models provide unique insights into diagnostics of complex diseases, engineering biological systems, and environment remediation.
Kinetics modeling has made steady progress over the last several years.
Examples include
a complete model of a human pathogen,
and over 1,000 models of gene circuits published in BioModels.

Unfortunately, the ability to develop kinetics models for biomedical applications is severely impaired by the lack of tools 
for model building. 
The premise of this course is that **tools and techniques from software engineering can be adapted to
biomedical modeling to accelerate innovation in biomedical modeling and its application**.
Here are a two examples. 
- Kinetics models tend to be large and redundant because of reaction combinatorics. 
Software engineering uses templates to handle repetitive information.
Something similar may be possible to represent reactions between complexes.
- Kinetics models typically require specifying a series of chemical reactions.
Surprisingly, there are no tools for static error checking of these reactions, 
such as verifying mass balance and charge balance. 
Software engineers use `linters` to do static error checking.
A "reaction linter" might do static error
checking for mass and charge balance.

The seminar is one initiative being undertaken
to develop a research agenda for the NIH Reproducibility Center recently funded in UW BioEngineering.
As a result, the class will be coordinated with
BIOE 599, a topics class in modeling biomedical systems.


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

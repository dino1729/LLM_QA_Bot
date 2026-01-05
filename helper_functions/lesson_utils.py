import logging
import os
import random
import re
from typing import Any, Dict, Tuple, Optional

from helper_functions.llm_client import get_client

logger = logging.getLogger(__name__)

# List of topics
topics = [
    # --- Cognitive Biases & Psychology (1/3) ---
    "What are the tenets of effective office politics?",
    "How can I communicate more effectively?",
    "How can anchoring bias be leveraged to set favorable terms in salary or deal negotiations?",
    "When is tit-for-tat with forgiveness more effective than strict reciprocity in office politics?",
    "What strategies can mitigate the negative effects of Conway's Law on modularity in IP design (e.g., reusable components)?",
    "What does cognitive bias research teach about better decision-making?",
    "How can the Ben Franklin Effect be leveraged to turn rivals into allies?",
    "How does the Self-Serving Bias protect our ego but hinder our growth?",
    "How does the Bandwagon Effect drive market bubbles and social trends?",
    "How does the Halo Effect cause us to irrationally trust attractive or charismatic leaders?",
    "How does the False Consensus Effect make us overestimate agreement with our beliefs?",
    "How does the Spotlight Effect make us overestimate how much others notice us?",
    "How does the Availability Heuristic skew our risk assessment based on recent news?",
    "How does Naïve Realism make us believe we see the world objectively while others are biased?",
    "How does the Barnum Effect explain the popularity of horoscopes and personality tests?",
    "How does the Dunning-Kruger Effect prevent incompetent people from recognizing their incompetence?",
    "How does Anchoring bias our decisions based on the first piece of information we receive?",
    "How does Automation Bias lead to over-reliance on systems and reduced vigilance?",
    "How does the Google Effect (Digital Amnesia) change how we store and retrieve information?",
    "How does Confirmation Bias filter out information that contradicts our beliefs?",
    "How does the Backfire Effect strengthen beliefs when they are challenged by evidence?",
    "How does the Third-Person Effect make us believe others are more influenced by media than we are?",
    "How does Belief Bias cause us to judge arguments by their conclusion rather than logic?",
    "How does Status Quo Bias make organizational change so difficult?",
    "How does the Sunk Cost Fallacy keep us in bad investments or relationships?",
    "How does the Gambler's Fallacy lead to incorrect predictions in random events?",
    "How does Zero-Risk Bias make us prefer eliminating small risks over reducing large ones?",
    "How does the Framing Effect change our decisions based on whether options are presented as gains or losses?",
    "How does Outgroup Homogeneity Bias make us see 'them' as all the same?",
    "How does Authority Bias lead to blind obedience in hierarchical structures?",

    # --- Cognitive Biases & Psychology (2/3) ---
    "How does the Placebo Effect demonstrate the power of belief on physical outcomes?",
    "How does Survivorship Bias distort our understanding of success by ignoring failures?",
    "How does the Zeigarnik Effect help us remember uncompleted tasks better than completed ones?",
    "How does the IKEA Effect make us overvalue things we helped create?",
    "How does the Bystander Effect reduce the likelihood of help in emergencies?",
    "How does the Clustering Illusion make us see patterns in random data?",
    "How does Pessimism Bias lead to overestimating the likelihood of negative outcomes?",
    "How does Optimism Bias lead to underestimating risks and poor planning?",
    "How does the Blind Spot Bias prevent us from seeing our own cognitive biases?",
    "How does the Recency Illusion make us think recent phenomena are new?",
    "How does the Primacy Effect make first impressions so enduring?",
    "How does the Recency Effect make us weigh the latest information most heavily?",
    "How does the Serial Position Effect influence what we remember from a list?",
    "How does the Planning Fallacy lead to consistent underestimation of time and cost?",
    "How does Pro-Innovation Bias lead to overlooking the downsides of new technology?",
    "How does Projection Bias make us assume our future selves will share our current tastes?",
    "How does Restraint Bias lead to overestimating our ability to control impulses?",
    "How does Consistency Bias make us rewrite our past to fit our current beliefs?",
    "How does Hindsight Bias make unpredictable events seem inevitable after the fact?",
    "How does Outcome Bias cause us to judge decisions by results rather than process?",
    "How does Impact Bias make us overestimate the duration of emotional reactions?",
    "How does Information Bias cause us to seek data that won't change our decision?",
    "How does the Ambiguity Effect make us avoid options with missing information?",
    "How does the Contrast Effect alter our perception of value when comparing options?",
    "How does Distinction Bias lead to over-optimizing for differences that don't matter in practice?",
    "How does the Endowment Effect make us overvalue what we own?",
    "How does the Illusion of Control make us believe we can influence random outcomes?",
    "How does the Illusion of Validity give us false confidence in our predictions?",
    "How does Normalcy Bias delay reaction to disasters?",
    "How does Observation Selection Bias make us notice things we are looking for?",

    # --- Cognitive Biases & Psychology (3/3) ---
    "How does the Ostrich Effect cause us to avoid negative financial information?",
    "How does the Rhyme-as-Reason Effect make rhyming statements seem more truthful?",
    "How does Social Desirability Bias affect survey responses and public behavior?",
    "How does Trait Ascription Bias make us view ourselves as variable but others as predictable?",
    "How does Unit Bias influence portion control and consumption?",
    "How does the Well-Traveled Road Effect make familiar routes seem shorter?",
    "How does the Women-are-Wonderful Effect influence positive stereotypes?",
    "How does the Worse-than-Average Effect occur in difficult tasks?",
    "How does Zero-Sum Bias make us see interactions as win-lose when they could be win-win?",
    "How does the Cheerleader Effect alter our perception of individuals in groups?",
    "How does the Fundamental Attribution Error distort our judgment of others' character vs. situation?",
    "How does the Semmelweis Reflex cause automatic rejection of new knowledge?",
    "How does System Justification make us defend the status quo even when disadvantaged?",
    "How does the Ultimate Attribution Error extend attribution error to groups?",

    # --- History & Civilizations (1/7) ---
    "What can I learn from Boyd, the fighter pilot who changed the art of war?",
    "How can history inform decision-making?",
    "What strategic principles can we learn from the Roman Empire's military campaigns?",
    "How did the Renaissance shift thinking in art, science, and humanism?",
    "What can leaders learn from the resilience of ancient civilizations like Egypt or Mesopotamia?",
    "How did the Industrial Revolution reshape economies and societies globally?",
    "What parallels exist between the fall of ancient empires and modern geopolitical tensions?",
    "How did the Cold War shape technological innovation and global alliances?",
    "How did ancient trade routes like the Silk Road foster globalization before the modern era?",
    "What strategies did historical figures use to overcome adversity and resistance?",
    "How can historical breakthroughs in science guide today's ethical decisions in tech?",
    "What are the pivotal lessons from major world revolutions (American, French, etc.)?",
    "How have ideas of human rights evolved through history and influenced modern policies?",
    "How did the ancient Greeks lay foundations for democracy, philosophy, and science?",
    "What can modern businesses learn from the strategies of historical empires?",
    "How did the printing press revolutionize knowledge dissemination and what parallels exist with today's internet?",
    "What lessons can leaders draw from the failures of historical leaders?",
    "How did ancient engineering feats like the Pyramids or Roman aqueducts influence modern engineering?",
    "What historical patterns repeat in modern politics and economics?",
    "What lessons in ethics can we draw from historical debates on slavery, suffrage, and civil rights?",
    "What can the rise and fall of ancient cities teach about sustainability and urban planning?",
    "How have pandemics throughout history influenced public health and societal change?",
    "What can the history of education systems teach about improving learning today?",
    "How did ancient law codes (like Hammurabi) influence modern legal systems?",
    "What lessons about tolerance and coexistence emerge from multiethnic empires?",
    "What are the strategic lessons from historical espionage and intelligence operations?",
    "How did the Scientific Revolution change humanity's relationship with nature?",
    "How did ancient democracies handle corruption, and what parallels exist today?",
    "What can we learn from the cultural renaissances across different civilizations?",
    "How did the invention of the internet compare to earlier communication revolutions?",

    # --- History & Civilizations (2/7) ---
    "How did ancient philosophers debate ethics, and how can those debates inform modern dilemmas?",
    "What can the history of medicine teach about innovation and patient care?",
    "What can the history of labor movements teach about modern workplace dynamics?",
    "How did ancient navigators use science to master the seas?",
    "What can we learn from the preservation and loss of ancient knowledge?",
    "How did the rise of nation-states transform Europe in the Middle Ages?",
    "What are the enduring lessons from the Roman Republic's transition to empire?",
    "How did Enlightenment thinkers influence the French and American revolutions?",
    "How did military alliances shape the outcomes of World Wars I and II?",
    "What can we learn from historical debates on free speech and censorship?",
    "How did ancient agricultural innovations support the growth of civilizations?",
    "What can business leaders learn from the successes and failures of ancient traders?",
    "What can leaders learn from wartime decision-making and crises?",
    "How did cultural revolutions shape modern music, literature, and art?",
    "What lessons can investors learn from historical market bubbles and crashes?",
    "How did the history of science shape today's research ethics and practices?",
    "How did the colonial independence movements succeed against empires?",
    "What lessons from ancient and modern engineers can solve today's infrastructural challenges?",
    "How do historical communication revolutions parallel today's social media landscape?",
    "How does the Roman Empire's infrastructure compare to modern megaprojects?",
    "What can we learn from history about balancing security and liberty?",
    "How can ancient philosophies guide modern debates on technology ethics?",
    "What can the history of global trade teach about modern supply chain resilience?",
    "How have societies historically responded to inequality, and what worked?",
    "What lessons about innovation emerge from the history of patents and intellectual property?",
    "How did women's roles evolve across different historical periods and cultures?",
    "What can we learn from the rise and fall of historical financial centers?",
    "How did post-war reconstruction efforts succeed or fail in rebuilding nations?",
    "What can the history of public health teach about vaccination and disease prevention?",
    "How did historical education reforms shape modern schooling?",

    # --- History & Civilizations (3/7) ---
    "How do historical debates about privacy and surveillance inform today's issues?",
    "What can the history of propaganda teach about media literacy today?",
    "What can leaders learn from historical explorations and risk-taking ventures?",
    "How did the ancient city-states balance rivalry and cooperation?",
    "How did historical scientific discoveries challenge prevailing worldviews?",
    "What lessons about resilience and innovation emerge from the history of technology?",
    "What can we learn from the history of philanthropy and social reform?",
    "What lessons about governance emerge from the failures of ancient regimes?",
    "How did historical migration patterns shape cultural identities and demographics?",
    "What can we learn from the history of urban planning and architecture?",
    "How did revolutionary ideas spread before modern communication tools?",
    "What can the history of religious reform teach about social change?",
    "What lessons about economic policy emerge from historical trade wars?",
    "How did historical leaders navigate crises and maintain legitimacy?",
    "What can the history of exploration teach about curiosity and discovery?",
    "How did historical thinkers grapple with the ethics of power and authority?",
    "What can we learn from the history of education in fostering critical thinking?",
    "How did historical debates on justice influence modern legal principles?",
    "What lessons from ancient democracy apply to modern civic engagement?",
    "How did historical rebellions shape the course of empires?",
    "What can the history of science communication teach about bridging experts and the public?",
    "How did historical advances in navigation and cartography change worldviews?",
    "What can we learn from the history of innovation hubs and creative cities?",
    "How did historical epidemics influence art, culture, and religion?",
    "How did historical financial crises reshape economies and regulations?",
    "What can the history of volunteerism and civic engagement teach today?",
    "How did shifts in trade routes impact the rise and fall of cities and empires?",
    "How did historical figures balance personal ambition with public service?",
    "What lessons from the fall of civilizations can inform modern resilience planning?",
    "How have concepts of war and peace evolved throughout history?",

    # --- History & Civilizations (4/7) ---
    "How did historical communities organize for mutual aid and support?",
    "What lessons emerge from the history of innovation during times of constraint?",
    "What can the history of media teach about the power of storytelling?",
    "How did historical thinkers approach the balance between faith and reason?",
    "How did historical societies handle dissent and protest?",
    "What can the history of mathematics teach about problem-solving approaches?",
    "How did historical cultures view and manage mental health?",
    "How did historical financial instruments develop and spread?",
    "What can we learn from the history of environmental stewardship?",
    "How did historical leaders cultivate legitimacy and trust?",
    "What lessons about collaboration emerge from large historical projects?",
    "How did historical societies adapt to climate and environmental changes?",
    "What can we learn from the history of scientific collaboration across borders?",
    "How did historical innovations in communication reshape power structures?",
    "What can we learn from the history of public discourse and debate?",
    "How did historical leaders manage information and intelligence?",
    "How did historical advances in transportation change economies and cultures?",
    "What can we learn from the history of technological ethics debates?",
    "How did historical societies foster innovation through education and patronage?",
    "What lessons emerge from historical examples of peaceful resistance?",
    "How did historical thinkers approach the concept of progress?",
    "What can we learn from the evolution of governance models across civilizations?",
    "How did historical leaders use symbolism and rituals to maintain power?",
    "How did historical cultures measure and value time?",
    "What can we learn from historical precedents of globalization?",
    "How did historical societies document and preserve knowledge?",
    "How did historical leaders balance local and central governance?",
    "What can we learn from the history of scientific instrumentation?",
    "How did historical debates on morality shape laws and customs?",
    "How did historical societies handle misinformation and rumor?",

    # --- History & Civilizations (5/7) ---
    "How did historical leaders manage succession and continuity?",
    "What lessons about resilience emerge from cultural renaissances?",
    "How did historical societies integrate new technologies into daily life?",
    "What can we learn from historical models of community governance?",
    "How did historical thinkers approach the relationship between humans and nature?",
    "How did historical societies encourage or suppress creativity?",
    "What can we learn from the history of measurement and standardization?",
    "How did historical leaders respond to technological disruption?",
    "How did historical societies view and manage risk?",
    "What lessons about communication emerge from historical rhetoric and persuasion?",
    "How did historical societies adapt to demographic shifts and migrations?",
    "What can we learn from the history of experimentation and scientific method?",
    "How did historical leaders use information networks to maintain control?",
    "How did historical societies construct and challenge social hierarchies?",
    "What can we learn from historical narratives of progress and decline?",
    "How did historical societies foster civic engagement and responsibility?",
    "What can we learn from the history of artistic and scientific patronage?",
    "How did historical leaders manage resource scarcity and abundance?",
    "What lessons about resilience emerge from historical recoveries after crises?",
    "How did historical societies view the role of technology in shaping destiny?",
    "What lessons about governance emerge from historical experiments in democracy?",
    "How did historical societies navigate cultural identity amid change?",
    "What can we learn from historical debates about justice and fairness?",
    "How did historical leaders build institutions that lasted?",
    "How did historical societies measure and value knowledge?",
    "What can we learn from historical approaches to balancing power among branches of government?",
    "How did historical societies cultivate wisdom across generations?",
    "How did historical leaders manage change during transformative periods?",
    "What lessons about collaboration emerge from historical alliances?",
    "What can we learn from historical precedents for balancing innovation with caution?",

    # --- History & Civilizations (6/7) ---
    "How did historical thinkers address the tension between individualism and collectivism?",
    "What lessons about resilience emerge from historical community rebuilding?",
    "How did historical societies institutionalize learning and knowledge sharing?",
    "How did historical leaders use narrative to inspire and mobilize?",
    "What lessons about governance emerge from historical federations and unions?",
    "How did historical societies handle rapid technological change?",
    "How did historical thinkers conceptualize the common good?",
    "How did historical societies foster innovation ecosystems?",
    "How did historical leaders cultivate adaptability and foresight?",
    "What lessons about communication emerge from historical diplomacy?",
    "How did historical societies measure success and prosperity?",
    "What can we learn from historical examples of interdisciplinary innovation?",
    "How did historical thinkers address the moral dimensions of progress?",
    "How did historical societies navigate competing priorities and trade-offs?",
    "What can we learn from historical practices that sustained cultural heritage?",
    "How did historical leaders use education as a strategic tool?",
    "How did historical societies view the responsibilities of the educated class?",
    "What can we learn from historical debates on rights and responsibilities?",
    "How did historical thinkers integrate philosophy with practical governance?",
    "What lessons about resilience emerge from historical adaptation to new eras?",
    "How did historical societies design systems to balance power and accountability?",
    "What can we learn from historical approaches to cultivating virtue and character?",
    "How did historical leaders manage diverse coalitions and interests?",
    "What lessons about innovation emerge from historical leaps in understanding?",
    "How did historical societies ensure continuity amid change?",
    "What can we learn from historical reflections on time, progress, and legacy?",
    "How did historical thinkers frame the purpose of education and knowledge?",
    "How did historical societies institutionalize wisdom traditions?",
    "What can we learn from historical explorations of justice, power, and ethics?",
    "What led to the decline of the Julio-Claudian dynasty?",

    # --- History & Civilizations (7/7) ---
    "How did Alexander integrate diverse cultures within his empire?",
    "How did Rome transition from Republic to Empire politically and culturally?",
    "How did the Founding Fathers balance federal and state power in the Constitution?",
    "What strategic decisions turned the tide in World War II's Pacific theater?",
    "How did railroads reshape economic geography during the Industrial Revolution?",
    "How did open-source movements change the trajectory of software innovation?",
    "How did Chanakya's Arthashastra guide statecraft and economics in ancient India?",
    "What factors led to the rise and fall of the Mughal Empire?",
    "What can software teams learn from agile practices in film production?",
    "What strategies helped women leaders navigate male-dominated fields historically?",
    "What lessons from past energy transitions apply to today's shift toward renewables?",
    "How did China's tributary system manage foreign relations during the Ming dynasty?",
    "What internal factors led to the Qing dynasty's struggles with Western powers?",
    "How did the ancient trade networks of the Silk Roads facilitate cultural exchange and technological diffusion?",
    "What does a Silk Roads perspective teach us about geopolitical power centers throughout history?",
    "What advanced civilizations existed in pre-Columbian Americas that challenge our historical narratives?",
    "How did the dual revolutions (French and Industrial) fundamentally reshape European society?",
    "What economic and social factors drove the revolutionary changes across Europe from 1789-1848?",
    "What administrative innovations allowed the Ottoman Empire to successfully govern a diverse, multi-ethnic state?",
    "How did the Ottoman Empire's position between East and West influence its cultural development?",
    "What internal factors contributed most significantly to the Roman Empire's decline?",
    "How did the rise of Christianity influence the political transformation of the Roman Empire?",
    "How did the Great Migration of African Americans transform both Northern and Southern American society?",
    "What personal stories from the Great Migration reveal about systemic racism and individual resilience?",
    "How did democratic Athens' political system influence its military decisions during the war?",

    # --- Leadership, Strategy & Negotiation (1/2) ---
    "How to apply the best game theory concepts in getting ahead in office poilitics?",
    "What are some best ways to play office politics?",
    "What are the top 8 ways the tit-for-tat strategy prevails in the repeated prisoner's dilemma, and how can these be applied to succeed in life and office politics?",
    "What are Chris Voss's key strategies from *Never Split the Difference* for hostage negotiations, and how can they apply to workplace conflicts?",
    "How can tactical empathy (e.g., labeling emotions, mirroring) improve outcomes in high-stakes negotiations?",
    "How do calibrated questions (e.g., *How am I supposed to do that?*) shift power dynamics in negotiations?",
    "What are 'Black Swan' tactics for uncovering hidden information in negotiations?",
    "How can Nash Equilibrium concepts guide decision-making in workplace collaborations?",
    "What real-world scenarios mimic the 'Chicken Game,' and how should you strategize in them?",
    "How does backward induction in game theory apply to long-term career or project planning?",
    "What are examples of zero-sum vs. positive-sum games in corporate negotiations?",
    "How can Boyd's OODA Loop (Observe, Orient, Decide, Act) improve decision-making under pressure?",
    "What game theory principles optimize resource allocation in cross-functional teams?",
    "How can the 'MAD' (Mutually Assured Destruction) concept deter adversarial behavior in workplaces?",
    "What tactics improve focus and deep work for engineers in noisy environments?",
    "How can engineers negotiate scope and timelines without harming relationships?",
    "What leadership lessons emerge from the lives of great explorers like Magellan or Zheng He?",
    "How did religious movements shape historical events and power dynamics?",
    "What leadership lessons emerge from historical figures like Napoleon or Alexander the Great?",
    "How did colonialism alter global power dynamics and cultural exchanges?",
    "How have military strategies evolved from ancient times to modern warfare?",
    "What can we learn about leadership from the founding of the United States?",
    "How did technological innovations like gunpowder or the steam engine change warfare and industry?",
    "What lessons from the Great Recession apply to financial risk management today?",
    "What can the history of civil rights movements teach about persistence and strategy?",
    "How have ideas about leadership evolved from monarchies to modern corporate structures?",
    "How did technological advances in warfare change geopolitical power balances?",
    "What lessons about leadership emerge from transformative historical moments?",
    "What lessons about leadership emerge from historical exploration expeditions?",
    "What lessons about crisis management emerge from historical disasters?",

    # --- Leadership, Strategy & Negotiation (2/2) ---
    "What lessons about negotiation emerge from historical treaties and alliances?",
    "What lessons about leadership emerge from historical reformers and revolutionaries?",
    "How did historical thinkers approach the ethics of power and leadership?",
    "What lessons about strategy emerge from historical conflicts and alliances?",
    "What lessons about leadership emerge from historical mentors and teachers?",
    "What can we learn from historical examples of ethical leadership?",
    "How did historical societies view the responsibilities of leadership?",
    "What lessons about strategy emerge from historical power shifts?",
    "What can we learn from historical models of leadership succession?",
    "What lessons about leadership emerge from historical statesmen and reformers?",
    "How did the leadership styles of Julius Caesar and Augustus differ?",
    "What leadership lessons emerge from Abraham Lincoln's Civil War strategy?",
    "How do Sun Tzu's principles apply to modern business competition?",
    "What modern lessons come from the OODA loop in fast-moving markets?",
    "What does Drucker's management philosophy advise for knowledge workers?",
    "What negotiation tactics did Ruth Bader Ginsburg use to build consensus on the Supreme Court?",
    "How did the Indian independence movement balance nonviolent resistance with political strategy?",
    "What leadership lessons come from Sardar Patel's role in unifying princely states?",
    "How did Admiral Nelson's tactics at Trafalgar depart from naval convention?",
    "What leadership lessons emerge from the Buddhist concept of the Middle Way?",
    "How did FDR's diplomatic strategy shape the post-WWII international order?",
    "What lessons does Roosevelt's leadership offer for modern global cooperation?",
    "What strategic and leadership lessons can be learned from Athens' and Sparta's conflict?",
    "How can you apply the strategy of making your opponent believe they are playing a different game entirely, so they provide exactly what you need to win the real prize?",

    # --- Personal Development & Productivity ---
    "How to be more empathetic?",
    "How can I seek the mentorship I want from key influential people",
    "How to be more persuasive, assertive, influential, impactful, engaging, inspiring, motivating, captivating and convincing in my communication?",
    "What non-verbal cues (e.g., tone, body language) most impact persuasive communication?",
    "How can I adapt my communication style to different personality types (e.g., assertive vs. analytical)?",
    "How do you balance assertiveness and empathy when delivering critical feedback?",
    "How do Schelling Points (focal points) help teams reach consensus without direct communication?",
    "How can Bayesian reasoning improve decision-making under uncertainty (e.g., mergers, market entry)?",
    "What patterns make technical presentations persuasive to non-technical stakeholders?",
    "What habits build credibility and trust with cross-functional partners?",
    "What does Leonardo's notebook practice teach about creative problem-solving?",
    "How does neuroplasticity evidence support lifelong learning strategies?",
    "How does utilitarianism differ from deontology in tough decision scenarios?",
    "What does virtue ethics suggest about character-building habits for leaders?",
    "What design decisions in early computer architecture still influence systems today?",
    "How does Groupthink stifle innovation and lead to catastrophic decisions?",
    "How does the Law of Triviality (Bike-Shedding) cause teams to focus on minor details over major issues?",
    "How does Identification with a group or figure bypass critical thinking?",

    # --- Philosophy & Ethics ---
    "How has the concept of liberty evolved from the Magna Carta to modern democracies?",
    "What can we learn from the moral and political philosophies of thinkers like Kant, Locke, or Rousseau?",
    "How did the Enlightenment change views on reason, science, and individual rights?",
    "How have ideas about citizenship and rights evolved over time?",
    "What can we learn from the evolution of legal rights and protections?",
    "How did Aristotle's Golden Mean shape Western ethics?",
    "What are the key differences between Stoicism and Epicureanism in handling adversity?",
    "What ethical frameworks guide responsible innovation in genetics?",
    "How did Buddhism respond to and reshape prevailing Hindu philosophies?",
    "What moral dilemmas faced scientists during the Manhattan Project, and how are they relevant today?",
    "How does Moral Luck influence our praise or blame of actions based on outcomes?",
    "How does the Just-World Hypothesis lead to rationalizing injustice?",

    # --- Science, Innovation & Technology ---
    "Give me suggestions to reduce using filler words when communicating highly technical topics?",
    "What is the 'Accusations Audit' technique, and how does it disarm resistance in adversarial conversations?",
    "How can active listening techniques improve conflict resolution in team settings?",
    "What are highly scalable ways for Dinesh to be a changemaker in the semi-conductor space now, and in the future?",
    "What can the space race teach us about competition, innovation, and national pride?",
    "How did transportation innovations like railways and automobiles reshape economies and cities?",
    "How did global conflicts catalyze innovation in science and medicine?",
    "What lessons about innovation emerge from cross-cultural exchanges?",
    "How did the Medici patronage system transform the arts and sciences in Florence?",
    "What banking innovations did the Medici create, and how did they shape modern finance?",
    "How did Leonardo's cross-disciplinary curiosity fuel his innovation?",
    "What logistical innovations enabled Alexander's rapid military campaigns?",
    "How did early computer pioneers balance vision with execution?",
    "How does Schumpeter's 'creative destruction' explain tech industry shifts?",
    "What factors make certain regions enduring hubs of innovation?",
    "What public health innovations emerged from industrial urbanization?",
    "What lessons from SpaceX's iterative testing apply to other industries?",
    "How will generative AI reshape creative professions over the next decade?",
    "How have antitrust approaches to tech monopolies evolved over time?",
    "What lessons from telecom regulation apply to today's internet platforms?",
    "How did the discovery of CRISPR reshape bioengineering possibilities?",
    "How do AI accelerator design choices impact total cost of ownership in data centers?",
    "How did the development of nuclear weapons transform the relationship between science and government?",

    # --- Economics, Business & Finance ---
    "How did maritime exploration transform economies and cultures in the Age of Discovery?",
    "How did the Great Depression shape modern economic policy and financial regulation?",
    "How did the transatlantic slave trade shape global economies and societies?",
    "What engineering tradeoffs shaped the design of the first microprocessors?",
    "How does supply chain resilience influence national security strategies?",
    "What cultural tradeoffs did Peter the Great make in westernizing Russia?",

    # --- Society, Culture & Education ---
    "How did the abolitionist movement succeed in changing hearts, minds, and laws?",
    "How did imperialism shape today's geopolitical borders and tensions?",
    "How did industrialization impact social structures and daily life?",
    "What can be learned from the resilience of cultures that endured conquests and disasters?",
    "How have concepts of freedom and equality evolved across cultures and eras?",
    "What lessons about diversity and inclusion emerge from multiethnic societies?",
    "How did RBG strategically select cases to advance gender equality law?",
    "How did Peter the Great's reforms modernize Russia's military and governance?",
    "How did indigenous American societies develop sophisticated agricultural and urban systems before European contact?",
    "How does In-Group Favoritism shape organizational culture and conflict?",
    "How does Stereotyping simplify social processing at the cost of accuracy?",
    "How does Social Proof drive behavior in uncertainty?",
    "How does Scarcity artificially increase perceived value?",
    "How does Reciprocity create a feeling of indebtedness in social exchange?",
    "How does the Weber-Fechner Law explain our perception of price differences?",

    # --- General Wisdom & Miscellaneous (1/2) ---
    "How are electric vehicles less harmful to the environment?",
    "How can I think clearly in adverse scenarios?",
    "When should you use the 'Late-Night FM DJ Voice' to de-escalate tension during disagreements?",
    "What storytelling frameworks make complex ideas more compelling during presentations?",
    "What are strategies for managing difficult conversations (e.g., layoffs, project failures) with grace?",
    "How can you architect low-power designs for battery-sensitive devices (DVFS, power gating)?",
    "What are best practices for cross-team design reviews to catch issues early?",
    "How do high-performing teams handle late-stage design changes without chaos?",
    "What are practical steps to reduce meeting overload while staying aligned?",
    "What is the best play to lead the future of semi-conductors in a world of AI?",
    "What can we learn about resilience from societies that rebuilt after disasters?",
    "What can the fall of the Berlin Wall teach about resilience and the power of people?",
    "What can leaders learn from the mistakes of the League of Nations?",
    "What can we learn from the cultural exchanges of the Islamic Golden Age?",
    "What can we learn from the cultural syncretism of regions like the Mediterranean?",
    "How have legal systems evolved to balance tradition and progress?",
    "What lessons about sustainability emerge from societies that managed natural resources well?",
    "How have scientific paradigms shifted over time, and what drove those changes?",
    "What lessons about resilience emerge from communities rebuilding after conflict?",
    "What factors enabled the Medici family to wield power without formal titles?",
    "What collaboration patterns in 'The Innovators' accelerated breakthroughs in computing?",
    "What infrastructure strategies allowed Rome to manage vast territories?",
    "How did codebreaking at Bletchley Park influence Allied success?",
    "How did the printing press disrupt information control in Europe?",
    "What Enlightenment ideals laid groundwork for modern democracy?",
    "How did Maxwell's equations unify electricity and magnetism conceptually?",
    "How did the Apollo program manage risk under extreme time pressure?",
    "How did Toyota's production system principles spread globally?",
    "How can feedback loops and leverage points improve organizational change efforts?",
    "What can chaos theory teach leaders about planning in uncertainty?",

    # --- General Wisdom & Miscellaneous (2/2) ---
    "What policies balance AI-driven productivity with worker well-being?",
    "How did the transition from vacuum tubes to transistors unlock new computing paradigms?",
    "What can modern leaders learn from the logistics of historic naval expeditions?",
    "How did jazz improvisation influence modern creativity frameworks?",
    "How did pioneers like Ada Lovelace and Grace Hopper expand computing's possibilities?",
    "How does the Curse of Knowledge make it difficult for experts to teach beginners?",
    "How does Defensive Attribution cause us to blame victims to protect our worldview?",
    "How does Cynicism prevent us from seeing genuine altruism or opportunity?",
    "How does Reactance cause people to resist when they feel their freedom is threatened?",
    "How does the Availability Cascade turn minor events into collective panics?",
    "How does Declinism make us view the past as better and the future as worse?",
    "How does Tachypsychia alter our perception of time during trauma or high stress?",
    "How does Suggestibility lead to false memories or altered perceptions?",
    "How does False Memory created by leading questions impact legal testimony?",
    "How does Cryptomnesia lead to unintentional plagiarism?",
    "How does the Peak-End Rule determine how we remember experiences?",
    "How does the Liking principle influence persuasion and compliance?",
    "How does Commitment and Consistency drive us to align actions with past statements?",
    "How does the Empathy Gap make it hard to predict our behavior in different emotional states?",
    "How does Hedonic Adaptation return us to a baseline of happiness despite changes?",
    "How does Loss Aversion make the pain of losing stronger than the joy of gaining?",
    "How does Neglect of Probability lead to fear of unlikely but dramatic events?",
    "How does Pareidolia cause us to see faces or patterns where none exist?",
    "How does Risk Compensation cause us to take more risks when we feel safer?",
    "How does Selective Perception filter our experience to match our expectations?",

]


_HARDWARE_SEMI_RE = re.compile(
    r"\b("
    r"semi[-\s]?conductor\w*|silicon|soc\b|asic\b|rtl\b|verilog|fpga|"
    r"microprocessor\w*|low[-\s]?power|dvfs|power gating|tape[-\s]?out|"
    r"ai accelerator\w*|data center\w*"
    r")\b",
    re.IGNORECASE,
)
_NEGOTIATION_RE = re.compile(
    r"\b("
    r"never split the difference|chris voss|hostage|tactical empathy|calibrated questions|"
    r"black swan|accusations audit"
    r")\b",
    re.IGNORECASE,
)
_NEGOTIATION_GENERAL_RE = re.compile(
    r"\b(negotiat\w*|bargain\w*|persuad\w*|deal\w*|consensus)\b",
    re.IGNORECASE,
)
_GAME_THEORY_RE = re.compile(
    r"\b("
    r"game theory|nash|equilibrium|chicken game|backward induction|zero-sum|positive-sum|"
    r"prisoner'?s dilemma|tit-for-tat|mutually assured destruction|\bmad\b|ooda loop|ooda"
    r")\b",
    re.IGNORECASE,
)
_COGNITIVE_RE = re.compile(
    r"\b("
    r"bias|biases|effect|effects|fallacy|heuristic|illusion|attribution|groupthink|"
    r"placebo|loss aversion|hedonic adaptation|empathy gap|peak-end|pareidolia|"
    r"selective perception|false memory|cryptomnesia|suggestibility|reactance|"
    r"curse of knowledge|declinism|semmelweis|system justification|na[iï]ve realism|"
    r"liking principle|commitment and consistency|availability cascade|cynicism|tachypsychia|"
    r"stereotyp\w*|scarcity|reciprocity|social proof|in-group favoritism|weber-fechner|"
    r"risk compensation|neglect of probability|just-world hypothesis|identification with a group"
    r")\b",
    re.IGNORECASE,
)
_PHILOSOPHY_RE = re.compile(
    r"\b("
    r"ethics?|moral|virtue|utilitarianism|deontology|stoicism|epicureanism|buddhism|"
    r"kant|locke|rousseau|aristotle|golden mean|manhattan project"
    r")\b",
    re.IGNORECASE,
)
_COMMUNICATION_RE = re.compile(
    r"\b("
    r"communicat\w*|present\w*|storytell\w*|listen\w*|filler words|"
    r"difficult conversations|de-?escalat\w*|tone|body language|non-verbal"
    r")\b",
    re.IGNORECASE,
)
_PRODUCTIVITY_RE = re.compile(
    r"\b("
    r"productiv\w*|habit\w*|focus\w*|deep work|time management|meeting\w*|"
    r"lifelong learning|neuroplasticity"
    r")\b",
    re.IGNORECASE,
)
_AI_TECH_FUTURE_RE = re.compile(
    r"\b("
    r"ai\b|artificial intelligence|generative ai|llm|bioengineering|crispr|nuclear weapons|"
    r"telecom regulation|antitrust|policy"
    r")\b",
    re.IGNORECASE,
)
_SYSTEMS_INNOVATION_RE = re.compile(
    r"\b("
    r"feedback loops?|leverage points?|chaos theory|creative destruction|open-source|"
    r"iterative testing|innov\w*|pioneer\w*|space race|toyota|maxwell|"
    r"transistor\w*|vacuum tubes?|railway\w*|automobil\w*|jazz|leonardo|apollo|alexander"
    r")\b",
    re.IGNORECASE,
)
_SOCIETY_CULTURE_RE = re.compile(
    r"\b("
    r"abolition\w*|freedom|equality|diversity|inclusion|indigenous|multiethnic|syncretism"
    r")\b",
    re.IGNORECASE,
)
_EMOTIONAL_RE = re.compile(
    r"\b(empath\w*|adversity|resilien\w*|grace|critical feedback)\b",
    re.IGNORECASE,
)
_LEADERSHIP_TEAMS_RE = re.compile(
    r"\b("
    r"leadership|leaders?|manage\w*|management|team\w*|culture|collaboration|"
    r"credibility|trust|cross-functional|design reviews|alignment|conflict resolution"
    r")\b",
    re.IGNORECASE,
)
_OFFICE_POLITICS_RE = re.compile(
    r"\b("
    r"office politics|workplace|corporate|power dynamics|stakeholder\w*|promotion|career|mentorship"
    r")\b",
    re.IGNORECASE,
)
_DECISION_MAKING_RE = re.compile(
    r"\b("
    r"decision-making|decision making|uncertainty|risk management|scenario planning|"
    r"bayesian|first principles|planning"
    r")\b",
    re.IGNORECASE,
)
_SUSTAINABILITY_RE = re.compile(
    r"\b(environment\w*|sustainab\w*|renewables?|electric vehicles|energy transitions?)\b",
    re.IGNORECASE,
)
_HISTORY_BASE_RE = re.compile(
    r"\b("
    r"histor\w*|ancient\w*|empire\w*|civilization\w*|rome\w*|roman\w*|greek\w*|"
    r"egypt\w*|mesopotamia\w*|renaissance\w*|industrial revolution|cold war|world war\w*|"
    r"berlin wall|league of nations|"
    r"magna carta|enlightenment|medici|islamic golden age|silk road|apollo program|"
    r"bletchley park|printing press|genghis|mongol|colonial|imperialism|recession|depression|"
    r"lincoln|caesar|augustus|sun tzu|athens|sparta|peter the great|rbg|"
    r"revolut\w*|dynast\w*|nation-state\w*|founding fathers|post-war|reconstruction|"
    r"age of discovery|great migration|slave trade|industrialization|scientific revolution|"
    r"boyd\b|julio-claudian|qing\b"
    r")\b",
    re.IGNORECASE,
)

_HISTORY_GROUPS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "History: Meta & Patterns",
        re.compile(
            r"\b(history inform|historical patterns|parallels exist|repeat in modern)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "History: War & Conflict",
        re.compile(
            r"\b(war\w*|militar\w*|battle\w*|warfare|conflict\w*|espionage|intelligence|"
            r"nav\w*|civil war|world wars?|wwi|wwii|gunpowder|treat\w*|trafalgar|"
            r"codebreaking|crisis)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "History: Governance & Law",
        re.compile(
            r"\b(democr\w*|govern\w*|citizenship|rights?|libert\w*|law\w*|legal\w*|justice\w*|"
            r"censor\w*|free speech|privacy|surveil\w*|propagand\w*|legitim\w*|corrupt\w*|"
            r"regime\w*|state\w*|empire\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "History: Economy & Trade",
        re.compile(
            r"\b(econom\w*|trade\w*|market\w*|finance\w*|bank\w*|bubble\w*|crash\w*|"
            r"depression|recession|supply chain\w*|labor\w*|industrial\w*|patent\w*|"
            r"intellectual property|merchant\w*|trader\w*|currency|tax\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "History: Science & Technology",
        re.compile(
            r"\b(science\w*|scientific\w*|technolog\w*|innov\w*|invent\w*|research\w*|"
            r"internet|communication\w*|navigation\w*|cartograph\w*|instrument\w*|"
            r"mathemat\w*|comput\w*|printing press|apollo|crispr|nuclear)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "History: Society & Movements",
        re.compile(
            r"\b(civil rights|abolition\w*|suffrag\w*|women\w*|inequal\w*|migration\w*|"
            r"demograph\w*|multiethnic|urban\w*|education reform\w*|philanthropy|"
            r"social reform\w*|peace mov\w*|nonviolent|indigenous)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "History: Culture & Ideas",
        re.compile(
            r"\b(culture\w*|art\w*|music\w*|literature\w*|renaissance\w*|relig\w*|"
            r"philosoph\w*|ethic\w*|faith\w*|reason\w*|moral\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "History: Health & Medicine",
        re.compile(
            r"\b(health\w*|medicin\w*|vaccin\w*|disease\w*|epidem\w*|pandem\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "History: Exploration & Discovery",
        re.compile(
            r"\b(explor\w*|discover\w*|voyage\w*|expedition\w*|age of discovery|silk road)\b",
            re.IGNORECASE,
        ),
    ),
)


def _topic_group(topic: str) -> str:
    if _HARDWARE_SEMI_RE.search(topic):
        return "Tech: Hardware & Semiconductors"
    if _NEGOTIATION_RE.search(topic):
        return "Leadership: Negotiation (Chris Voss)"
    if _NEGOTIATION_GENERAL_RE.search(topic):
        return "Leadership: Negotiation & Persuasion"
    if _GAME_THEORY_RE.search(topic):
        return "Leadership: Game Theory & Strategy"
    if _PHILOSOPHY_RE.search(topic):
        return "Philosophy & Ethics"
    if _AI_TECH_FUTURE_RE.search(topic):
        return "Tech: AI, Regulation & Future"
    if _SUSTAINABILITY_RE.search(topic):
        return "Society: Sustainability & Environment"

    is_history = bool(_HISTORY_BASE_RE.search(topic))
    if is_history:
        for group_name, pattern in _HISTORY_GROUPS:
            if pattern.search(topic):
                return group_name
        return "History: Other"

    if _SYSTEMS_INNOVATION_RE.search(topic):
        return "Tech: Innovation & Systems Thinking"
    if _SOCIETY_CULTURE_RE.search(topic):
        return "Society: Culture, Equality & Inclusion"
    if _EMOTIONAL_RE.search(topic):
        return "Skills: Emotional Intelligence & Resilience"
    if _OFFICE_POLITICS_RE.search(topic):
        return "Leadership: Office Politics & Career"
    if _LEADERSHIP_TEAMS_RE.search(topic):
        return "Leadership: Management & Teams"
    if _DECISION_MAKING_RE.search(topic):
        return "Skills: Decision-Making & Reasoning"
    if _COMMUNICATION_RE.search(topic):
        return "Skills: Communication & Influence"
    if _PRODUCTIVITY_RE.search(topic):
        return "Skills: Productivity & Focus"
    if _COGNITIVE_RE.search(topic):
        return "Cognitive Biases & Psychology"

    return "Other"


def get_topics_by_group(topic_list: Optional[list[str]] = None) -> Dict[str, list[str]]:
    """
    Return topics grouped into finer-grained buckets for easier curation.

    Notes:
    - Each topic is assigned to exactly one group (first-match priority).
    - Grouping is heuristic and intended for pruning/organization.
    """
    topic_list = topic_list if topic_list is not None else topics
    grouped: Dict[str, list[str]] = {}
    for topic in topic_list:
        grouped.setdefault(_topic_group(topic), []).append(topic)
    return grouped


TOPICS_BY_GROUP = get_topics_by_group()

# List of personalities
personalities = [
    "Chanakya", "Sun Tzu", "Machiavelli", "Leonardo da Vinci", "Socrates", "Plato", "Aristotle",
    "Confucius", "Marcus Aurelius", "Friedrich Nietzsche", "Carl Jung", "Sigmund Freud",
    "Winston Churchill", "Abraham Lincoln", "Mahatma Gandhi", "Martin Luther King Jr.", "Nelson Mandela",
    "Albert Einstein", "Isaac Newton", "Marie Curie", "Stephen Hawking", "Richard Feynman", "Nikola Tesla",
    "Galileo Galilei", "James Clerk Maxwell", "Charles Darwin",
    "Alan Turing", "Claude Shannon", "Ada Lovelace", "Grace Hopper", "Tim Berners-Lee",
    "Linus Torvalds", "Guido van Rossum", "Dennis Ritchie",
    "Bill Gates", "Steve Jobs", "Elon Musk", "Jeff Bezos", "Satya Nadella", "Tim Cook",
    "Lisa Su", "Larry Page", "Sergey Brin", "Mark Zuckerberg", "Jensen Huang",
    "Gordon Moore", "Robert Noyce", "Andy Grove"
]


def parse_lesson_to_dict(lesson_text: str, topic: str = "") -> Dict[str, Any]:
    """
    Parse lesson text with [KEY INSIGHT], [HISTORICAL], [APPLICATION] markers
    into a structured dictionary.
    """
    result = {
        "topic": topic,
        "key_insight": "",
        "historical": "",
        "application": "",
        "raw_text": lesson_text,
    }

    if not lesson_text:
        return result

    key_insight_match = re.search(
        r"\[KEY INSIGHT\]\s*(.*?)(?=\[HISTORICAL\]|\[APPLICATION\]|$)",
        lesson_text,
        re.DOTALL | re.IGNORECASE,
    )
    historical_match = re.search(
        r"\[HISTORICAL\]\s*(.*?)(?=\[KEY INSIGHT\]|\[APPLICATION\]|$)",
        lesson_text,
        re.DOTALL | re.IGNORECASE,
    )
    application_match = re.search(
        r"\[APPLICATION\]\s*(.*?)(?=\[KEY INSIGHT\]|\[HISTORICAL\]|$)",
        lesson_text,
        re.DOTALL | re.IGNORECASE,
    )

    if key_insight_match and key_insight_match.group(1).strip():
        result["key_insight"] = key_insight_match.group(1).strip()
    else:
        first_marker_pos = len(lesson_text)
        for marker in [r"\[HISTORICAL\]", r"\[APPLICATION\]"]:
            match = re.search(marker, lesson_text, re.IGNORECASE)
            if match and match.start() < first_marker_pos:
                first_marker_pos = match.start()
        if first_marker_pos > 0:
            pre_text = lesson_text[:first_marker_pos].strip()
            if len(pre_text) > 30:
                result["key_insight"] = pre_text
        if not result["key_insight"]:
            result["key_insight"] = (
                "Timeless principles reveal themselves through historical patterns - "
                "understanding the past illuminates the path forward."
            )

    if historical_match and historical_match.group(1).strip():
        result["historical"] = historical_match.group(1).strip()

    if application_match and application_match.group(1).strip():
        result["application"] = application_match.group(1).strip()

    return result


def get_random_personality():
    used_personalities_file = "used_personalities.txt"

    if os.path.exists(used_personalities_file):
        with open(used_personalities_file, "r") as file:
            used_personalities = file.read().splitlines()
    else:
        used_personalities = []

    unused_personalities = list(set(personalities) - set(used_personalities))

    if not unused_personalities:
        unused_personalities = personalities.copy()
        used_personalities = []

    personality = random.choice(unused_personalities)
    used_personalities.append(personality)

    with open(used_personalities_file, "w") as file:
        for used_personality in used_personalities:
            file.write(f"{used_personality}\n")

    return personality


def get_random_topic():
    used_topics_file = "used_topics.txt"

    if os.path.exists(used_topics_file):
        with open(used_topics_file, "r") as file:
            used_topics = file.read().splitlines()
    else:
        used_topics = []

    unused_topics = list(set(topics) - set(used_topics))

    if not unused_topics:
        unused_topics = topics.copy()
        used_topics = []

    topic = random.choice(unused_topics)
    used_topics.append(topic)

    with open(used_topics_file, "w") as file:
        for used_topic in used_topics:
            file.write(f"{used_topic}\n")

    return topic


def get_random_lesson(llm_provider: str, model_tier: str) -> Tuple[str, str]:
    """
    Generate a comprehensive daily lesson using LLM's knowledge.
    Returns (topic, lesson_text).
    """
    topic = get_random_topic()
    prompt = f"""Topic: {topic}

YOU MUST START YOUR RESPONSE WITH "[KEY INSIGHT]" - this is mandatory.

Generate wisdom in EXACTLY this structured format:

[KEY INSIGHT]
Write 1-2 powerful sentences capturing the core wisdom about this topic.

[HISTORICAL]
Share ONE brief historical anecdote (2-3 sentences) that illustrates this principle.

[APPLICATION]
Write 1-2 sentences on how this applies to modern engineering or leadership.

CRITICAL RULES:
1. Your FIRST line MUST be "[KEY INSIGHT]" followed by your insight
2. Then "[HISTORICAL]" followed by your historical example
3. Then "[APPLICATION]" followed by your application
4. Keep each section brief - this is for a newsletter
5. Do NOT skip any section - all three are required"""

    lesson_learned = generate_lesson_response(prompt, llm_provider, model_tier=model_tier)
    return topic, lesson_learned


def generate_lesson_response(user_message: str, llm_provider: str, model_tier: str) -> str:
    """
    Generate a comprehensive lesson/learning content with historical context.
    """
    logger.info("Generating lesson for topic: %s...", user_message[:100])
    client = get_client(provider=llm_provider, model_tier=model_tier)

    syspromptmessage = """You are EDITH, an expert teacher helping Dinesh master timeless principles of success.

MANDATORY FORMAT - YOUR RESPONSE MUST LOOK EXACTLY LIKE THIS:
[KEY INSIGHT]
<your key insight here - 1-2 sentences>

[HISTORICAL]
<your historical example here - 2-3 sentences>

[APPLICATION]
<your application here - 1-2 sentences>

RULES:
1. START with "[KEY INSIGHT]" on the very first line - no exceptions
2. Include ALL THREE sections in order: KEY INSIGHT, HISTORICAL, APPLICATION
3. Keep each section brief (1-3 sentences)
4. NO introductions, NO meta-commentary - just the formatted content

Context: Dinesh is a Senior Analog Circuit Design Engineer who values first principles thinking."""

    conversation = [
        {"role": "system", "content": syspromptmessage},
        {"role": "user", "content": user_message},
    ]

    try:
        message = client.chat_completion(
            messages=conversation,
            max_tokens=4000,
            temperature=0.65,
        )

        logger.debug(
            "Raw LLM response for lesson (first 300 chars): %s",
            message[:300] if message else "EMPTY",
        )
        logger.debug("Full raw response length: %s", len(message) if message else 0)

        if not message or len(message.strip()) == 0:
            logger.warning("LLM returned empty response for lesson generation")
            return _get_fallback_lesson()

        cleaned = message.strip()

        reasoning_patterns = [
            "the user wants", "user wants", "user says", "user is asking",
            "we need to", "we need", "we must", "let me provide", "let me",
            "here's", "here is", "i'll provide", "i'll", "i will",
            "task:", "topic:", "provide a", "generate", "create a lesson",
            "write a lesson", "this lesson", "this response", "my response",
            "to answer this", "in response", "based on"
        ]

        lines = cleaned.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not any(
                pattern in line.lower() for pattern in reasoning_patterns
            ):
                start_idx = i
                break
        cleaned = "\n".join(lines[start_idx:]).strip()

        if cleaned.lower().startswith("assistant:"):
            cleaned = cleaned[len("assistant:") :].strip()
        if cleaned.lower().startswith("assistant"):
            cleaned = cleaned[len("assistant") :].strip()
        if cleaned.lower().startswith("edith"):
            cleaned = cleaned[len("edith") :].strip()

        for marker in ["[KEY INSIGHT]", "[HISTORICAL]", "[APPLICATION]"]:
            if marker.lower() not in cleaned.lower():
                logger.warning("Missing required marker (%s), using fallback", marker)
                return _get_fallback_lesson()

        reasoning_patterns += [
            "reasoning process",
            "let's think step by step",
            "analysis",
            "let me think",
            "i will",
        ]
        for pattern in reasoning_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"(?m)^\s*(analysis|reasoning):\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(
            r"(?m)^\s*(thinking|thought process):\s*", "", cleaned, flags=re.I
        )
        cleaned = cleaned.strip()

        if cleaned.lower().startswith(("[key insight", "key insight")) is False:
            if "[KEY INSIGHT]" not in cleaned:
                cleaned = "[KEY INSIGHT]\n" + cleaned

        first_para = cleaned.split("\n\n")[0] if "\n\n" in cleaned else cleaned[:250]
        if any(pattern in first_para.lower() for pattern in reasoning_patterns):
            logger.debug("Found reasoning patterns in first paragraph, using second paragraph")
            paragraphs = cleaned.split("\n\n")
            if len(paragraphs) > 1:
                cleaned = "\n\n".join(paragraphs[1:]).strip()

        if cleaned and len(cleaned) > 200:
            logger.info("Successfully generated lesson (%s chars)", len(cleaned))
            return cleaned

        logger.warning(
            "Generated lesson too short (%s chars), using fallback",
            len(cleaned) if cleaned else 0,
        )
        return _get_fallback_lesson()

    except Exception as e:
        logger.error("Error generating lesson: %s", e, exc_info=True)
        return _get_fallback_lesson()


def _get_fallback_lesson():
    """Return a fallback lesson when generation fails"""
    fallback = """The pursuit of mastery requires understanding that excellence is not a destination but a continuous journey. Ancient philosophers like Aristotle understood this, coining the term "eudaimonia" to describe the flourishing that comes from living up to one's potential through disciplined practice.

Consider the example of Leonardo da Vinci, who kept detailed notebooks throughout his life, documenting not just his artistic techniques but his observations of nature, engineering concepts, and philosophical musings. This habit of systematic learning and documentation allowed him to make connections across domains that others missed.

For modern engineers and leaders, this translates to three practical principles: First, maintain a learning system - whether notebooks, digital tools, or structured reflection time. Second, seek cross-domain knowledge, as innovation often happens at the intersection of fields. Third, embrace deliberate practice over mere repetition, focusing on the areas where improvement is most needed.

The historical parallel to the Roman aqueduct engineers is instructive: they built systems that lasted millennia not through revolutionary innovation alone, but through meticulous attention to fundamentals, redundancy in design, and deep understanding of the materials and forces they worked with."""

    logger.info("Using fallback lesson content")
    return fallback

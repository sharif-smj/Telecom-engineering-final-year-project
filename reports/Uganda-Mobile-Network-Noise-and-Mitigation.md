# State of the Ugandan

# Telecommunications Infrastructure: A

# Comprehensive Analysis of Network

# Technologies, Coverage Dynamics, and

# Signal Integrity

## 1. Executive Introduction: The Digital Architecture of

## Uganda

The telecommunications sector in Uganda stands at a critical inflection point in the
mid-2020s, characterized by a rapid, capital-intensive divergence between legacy
infrastructure preservation and next-generation network deployment. As of late 2024 and
entering 2025, the sector is defined not merely by the expansion of coverage metrics—which
have reached impressive saturation points in terms of population reach—but by the intricate,
often invisible struggle to maintain signal integrity against a rising tide of spectral noise and
environmental interference.

The market is fundamentally a duopolistic competition between MTN Uganda and Airtel
Uganda, both of whom have transitioned from pure connectivity providers to central pillars of
the national financial and social infrastructure. This evolution is underpinned by a robust
regulatory framework enforced by the Uganda Communications Commission (UCC), which
manages a complex ecosystem of licensed operators, frequency allocations, and compliance
mandates. The sector has witnessed a surge in telephone subscriptions to 33.2 million and
broadband connections rising to 23.7 million.^1 However, these aggregate numbers mask
significant disparities in the quality of service (QoS) experienced across different districts,
influenced by topography, economic density, and technically complex interference
phenomena.

This report provides an exhaustive technical and geospatial audit of the mobile network
landscape in Uganda. It dissects the deployment of 2G, 3G, 4G, and 5G technologies across
the country's diverse districts, analyzes the specific electromagnetic and environmental
interference vectors degrading network performance, and evaluates the multi-layered
mitigation strategies currently being employed by operators and regulators to secure the
nation's digital future.


## 2. Technological Stratification: The Coexistence of

## Legacy and Frontier Networks

Unlike markets in the Global North that are aggressively sunsetting legacy networks to free up
spectrum, Uganda’s mobile network architecture resembles a complex geological formation
where distinct technology generations coexist, each serving a vital, non-redundant economic
function. The "layer-cake" network topology is necessitated by the stark economic divide
between urban centers, where 5G adoption is driven by enterprise needs, and rural
hinterlands, where 2G remains the lifeline for voice and USSD-based mobile money
transactions.

### 2.1 The Enduring Utility of 2G and 3G Layers

Despite the global narrative of 2G decommissioning, the technology remains the bedrock of
Ugandan connectivity. There is currently no immediate regulatory roadmap for a 2G sunset,
primarily because a significant proportion of the user base relies on feature phones.

```
● 2G (GSM/EDGE): This layer provides near-ubiquitous voice coverage and facilitates the
USSD protocol essential for Mobile Money (MoMo) platforms. In districts with rugged
terrain or low population density, 2G on the 900 MHz band is often the only available
signal due to its superior propagation characteristics compared to higher frequencies.
● 3G (UMTS/HSPA+): While 4G has surpassed 3G in efficiency, 3G networks act as a
critical failover for data continuity. However, operators are actively engaging in spectrum
refarming—reallocating 2100 MHz spectrum previously dedicated to 3G to bolster 4G
capacity. The UCC’s market reports indicate that while data consumption is pivoting to
4G, 3G retains a massive footprint for voice and legacy data devices.^1
```
### 2.2 The 4G LTE Standard: The Current Workhorse

Long Term Evolution (4G) has established itself as the primary carrier of digital traffic in
Uganda. The aggressive rollout of 4G infrastructure has been the focal point of capital
expenditure (CAPEX) for both major operators over the last five years.

```
● Population Coverage: The UCC reports a landmark achievement of 96% 4G population
coverage as of 2024.^2 This statistic places Uganda significantly ahead of the regional
East African average of 78% and the continental African average of 81%.
● Spectrum Utilization: Operators utilize a mix of low-band (800 MHz) for wide-area rural
coverage and mid-band (1800 MHz, 2600 MHz) for urban capacity. The assignment of
the 800 MHz "digital dividend" spectrum has been instrumental in penetrating buildings
and covering the expansive, sparsely populated northern districts.^3
● The Usage Gap: A critical paradox exists where coverage is nearly universal, yet
adoption lags. The "Usage Gap" is estimated at 75%, meaning three out of four Ugandans
living within the footprint of a 4G broadband network do not utilize it, primarily due to the
```

```
cost of 4G-enabled handsets and digital literacy barriers.^2
```
### 2.3 The 5G New Radio (NR) Frontier

The deployment of 5G represents the cutting edge of network engineering in Uganda, moving
rapidly from pilot phases in 2023 to commercial densification in 2024 and 2025.

```
● Commercial Deployment: Airtel Uganda has confirmed the operation of 365 live 5G
sites across the country.^5 Similarly, MTN Uganda reported a massive surge in 5G
coverage from a negligible 0.3% to 15.3% within a single fiscal year, signaling a fierce
infrastructure race.^7
● Spectrum Allocation: The UCC successfully executed a multi-band spectrum award in
mid-2023, granting operators access to the 700 MHz, 800 MHz, 2.3 GHz, 2.6 GHz, and
the critical 3.5 GHz (C-Band) frequencies.^3 The 3.5 GHz band serves as the capacity layer
for urban centers, while the 700 MHz band is earmarked to eventually extend 5G services
to rural areas, leveraging its long-range propagation capabilities.
● Strategic Use Cases: The current rollout strategy is described as "ahead of need,"
driven less by consumer handset demand and more by Fixed Wireless Access (FWA)
solutions for homes and businesses (WakaNet 5G), and industrial applications in the
oil-rich Albertine region.^5
```
## 3. Geospatial Analysis: District Coverage and Network

## Distribution

The distribution of network technology in Uganda is not uniform. It follows a distinct logic
dictated by economic corridors, topography, and recent government interventions through
the National Backbone Infrastructure (NBI). The following analysis breaks down coverage by
region and specific districts, synthesizing data from coverage maps, operator
announcements, and regulatory audits.

### 3.1 Central Region: The High-Density Core

The "Greater Kampala Metropolitan Area" (GKMA) and its surrounding districts enjoy
near-saturation coverage of 4G and the highest density of 5G nodes. This region generates
the bulk of data traffic and revenue.

```
● Key Districts: Kampala, Wakiso, Mukono, Entebbe, Kayunga, Mityana, Luwero,
Nakasongola, Buikwe (Njeru).
● Granular Coverage Details:
○ Kampala & Wakiso: These districts possess a dense mesh of 4G and 5G sites.
Independent mapping data confirms strong signal presence in suburbs such as
Njeru, Namasuba, Kireka, Bweyogerere, and Entebbe.^9
○ The Commuter Belt: As urbanization expands outwards, operators have prioritized
```

```
towns that serve as dormitories for the capital. 5G and high-capacity 4G are live in
Lugazi, Wobulenzi, and Bombo (Luwero) , catering to the high daily commuter
traffic.^9
○ Mubende & Kiboga: Located to the west of the capital, these districts act as
gateways. They are fully covered by 4G, with MTN and Airtel explicitly listing
Mubende and Kiboga as key coverage zones for their advanced networks.^9
● Quality of Service: Despite the high density of sites, the Central region suffers from
significant interference-related congestion (discussed in Chapter 5), leading to high
latency during peak hours.^11
```
### 3.2 Western Region: The Industrial and Tourism Axis

The Western region is a strategic priority due to the oil and gas sector in the Albertine Graben
and the high-value tourism circuits.

```
● Key Districts: Mbarara, Hoima, Masindi, Fort Portal (Kabarole), Kasese, Kabale, Kisoro,
Ntungamo, Kiryandongo.
● Strategic Nodes:
○ Hoima & Masindi: These districts are the epicenter of Uganda’s nascent oil industry.
Consequently, they have been prioritized for 5G deployment to support industrial IoT
and enterprise connectivity for oil majors.^5 UCC QoS reports indicate exceptionally
high call setup success rates in Masindi (97.4%), reflecting robust infrastructure.^12
○ Mbarara: As the primary commercial hub of the west, Mbarara City is fully
5G-enabled, with coverage extending to satellite towns like Bwizibwera.^13
○ Kabale & Kisoro: These mountainous border districts present severe topographical
challenges (signal diffraction). While urban centers have 4G, rural connectivity relies
heavily on the 800 MHz band. Airtel has confirmed 5G sites in Kabale , a significant
engineering feat given the terrain.^5
```
### 3.3 Northern Region: Post-Conflict Reconstruction and Digital

### Integration

Historically the most underserved region, the North is witnessing the fastest rate of new
infrastructure deployment, driven by government mandates to integrate the region into the
digital economy.

```
● Key Districts: Gulu, Lira, Arua, Kitgum, Moroto (Karamoja sub-region), Adjumani, Yumbe.
● Technology Status:
○ Urban Hubs: Gulu and Lira are now designated as 5G cities, with Airtel and MTN
deploying sites to support the growing commercial activity and the university
populations in these towns.^5
○ West Nile: Arua is a critical connectivity node bordering the DRC, with strong 4G
coverage and emerging 5G zones.^13
○ Karamoja Transformation: The most significant development in 2024 was the
```

```
launch of Phase V of the National Backbone Infrastructure (NBI) in Moroto. This
project has brought high-speed fiber connectivity to a region previously reliant on
unstable microwave links, fundamentally altering the connectivity landscape for
districts like Moroto, Napak, and Nakapiripirit.^14
```
### 3.4 Eastern Region: The Trade Corridor

The East is anchored by the trade route connecting landlocked Uganda to the port of
Mombasa.

```
● Key Districts: Jinja, Mbale, Soroti, Tororo, Busia, Iganga, Mayuge, Kapchorwa.
● Coverage Dynamics:
○ Jinja & Mbale: These cities are fully 5G operational. The industrial base in Jinja and
the commercial density in Mbale drive this investment.^5
○ Soroti: Coverage maps indicate solid 4G presence, with the town serving as a
connectivity hub for the Teso sub-region.^13
○ Border Towns: Busia and Tororo have high coverage density due to cross-border
trade, though they are also hotspots for cross-border signal interference.^15
```
**Table 1: Mobile Network Technology Status by Key District**

```
District Region 4G LTE Status 5G NR Status Strategic
Drivers
```
```
Kampala Central Ubiquitous Dense Political/Comm
ercial Capital
```
```
Wakiso Central Ubiquitous Dense Residential/Co
mmuter Hub
```
```
Mbarara Western High Live Regional
Commerce
```
```
Hoima Western High Live Oil & Gas
Industry
```
```
Gulu Northern High Live Regional
Hub/NGOs
```
```
Jinja Eastern High Live Industrial/Touri
sm
```

```
Mbale Eastern High Live Trade/Density
```
```
Moroto Northern Improving Planned NBI Phase V /
Gov Services
```
```
Kabale Western High (Urban) Live Border
Trade/Tourism
```
```
Masindi Western High Partial Oil Sector
Support
```
```
Lira Northern High Live Agro-processi
ng/Trade
```
## 4. The National Backbone Infrastructure (NBI): The

## Silent Enabler

No discussion of mobile coverage is complete without analyzing the fiber optic backbone that
backhauls data from towers to the core network. The National Information Technology
Authority (NITA-U) has been aggressively expanding the National Backbone Infrastructure
(NBI), which now spans over 4,300 kilometers.^14

### 4.1 Phase V Expansion: Closing the Ring

The recently completed Phase V of the NBI is a game-changer for the Northern and
Northeastern regions. Funded by the EXIM Bank of China, this phase extended connectivity to
the Karamoja sub-region (Moroto) and other "black spot" areas.^14

```
● Impact: By connecting district headquarters to the national fiber ring, NITA-U has
reduced the reliance on expensive and weather-susceptible satellite or microwave
backhaul. This lowers the operating expenditure (OPEX) for private telcos (MTN/Airtel),
incentivizing them to upgrade rural towers from 2G to 4G/5G.
● Kanungu Extension: In late 2024, the backbone was further extended to Kanungu
District , ensuring that deep rural areas in the southwest are integrated into the
high-speed network.^16
```
### 4.2 The "Last Mile" Challenge

While the backbone is extensive, the "last mile"—the connection from the fiber node to the
specific cell tower or home—remains a bottleneck. The government has secured funding for


the Uganda Digital Acceleration Project (UDAP) to extend this last-mile connectivity to 2,
sites, further densifying the network in sub-counties.^17

## 5. Technical Analysis of Noise and Signal Interference

While coverage maps suggest a connected nation, the user experience is often degraded by a
complex, invisible enemy: spectral interference. In Uganda, "noise" is not just a random static;
it is a structural problem driven by illegal equipment, unregulated broadcasting, and the
physics of the equatorial climate.

### 5.1 The "Booster" Menace: Wideband Impulse Noise

The most pervasive source of interference in Uganda is the illegal use of GSM/LTE signal
repeaters, colloquially known as "boosters."

```
● Mechanism of Interference: Residents in areas with poor indoor signal buy cheap,
unregulated bi-directional amplifiers. These devices often lack proper filtering and
isolation. They amplify the noise floor along with the signal, and crucially, they frequently
enter a state of oscillation. When a booster oscillates, it acts as a high-power jammer,
broadcasting a loud "screech" across the entire uplink frequency band.^18
● The "Near-Far" Problem: This creates a classic "Near-Far" imbalance at the base
station. The tower’s receiver becomes desensitized (deafened) by the high-power noise
from the illegal booster, preventing it from detecting the weaker signals from legitimate
mobile phones in the cell. The result is dropped calls and plummeted data throughput for
hundreds of users in the vicinity of a single illegal device.^19
● Regulatory Response: The UCC has criminalized the possession of these devices and
conducts raids to confiscate them. However, their proliferation remains a challenge due
to porous borders and high demand.^20
```
### 5.2 Spectral Pollution from "Bizindaalo" and Illegal Broadcasters

A unique interference vector in Uganda is the unregulated audio broadcasting sector.

```
● The Source: Unauthorized community radios, known as "Bizindaalo" (megaphones), and
illegal FM stations operate without technical oversight. These stations often use cheap
transmitters with poor harmonic filtering.^22
● Harmonic Distortion: A transmitter operating at 100 MHz with poor filtering can
generate strong harmonics at 200 MHz, 300 MHz, and up into the cellular bands
(800/900 MHz). Furthermore, Intermodulation Products —ghost signals created when
two transmitter frequencies mix in a non-linear device (like a rusty tower bolt)—can land
directly in the sensitive uplink bands of mobile operators.^21
● Impact: This raises the thermal noise floor of the network. For LTE and 5G, which rely on
high Signal-to-Noise Ratios (SNR) to achieve high modulation schemes (like 64-QAM or
```

```
256-QAM), this increased noise floor forces the network to drop to lower, slower
modulation schemes (like QPSK), drastically reducing internet speeds for users.^24
```
### 5.3 Environmental Attenuation: The Physics of Rain Fade

Uganda’s location on the equator introduces severe atmospheric interference known as "Rain
Fade," particularly affecting high-frequency backhaul and 5G.

```
● The Physics: Rain attenuation occurs when the wavelength of the radio signal is
comparable to the size of the raindrops. In Uganda’s intense tropical storms, rainfall rates
often exceed 100mm/h. This causes absorption and scattering of electromagnetic waves,
particularly in the Ku (12-18 GHz), K (18-27 GHz), and Ka (26-40 GHz) bands used for
microwave backhaul and future 5G mmWave.^26
● Network Availability: A microwave link designed with a "fade margin" of 20dB might still
be overwhelmed by a tropical downpour causing 30dB of attenuation. This leads to the
link disconnecting entirely. With many rural towers still reliant on microwave backhaul
(due to lack of fiber), heavy rains frequently cause localized network blackouts even
when power is available.^27
● 5G Implications: As Uganda moves toward higher 5G frequencies (C-Band and beyond),
the effective cell radius will shrink dramatically during rain, requiring much denser tower
placement to maintain coverage.^29
```
### 5.4 Cross-Border Signal Interference

In border districts, sovereign spectrum boundaries are often disrespected by the physics of
radio propagation.

```
● The Phenomenon: Signals from Kenyan, Rwandan, or South Sudanese towers often spill
over into Uganda. If a Ugandan operator and a neighbor use the same frequency block
(Co-Channel Interference), users at the border experience severe degradation. Their
phones struggle to lock onto the "home" network, often roaming inadvertently onto the
foreign network.^30
● Affected Areas: This is particularly acute in Busia, Tororo (Kenya border), and Kabale,
Kisoro (Rwanda border).
● One Network Area (ONA): The East African Community is working to harmonize these
frequencies and enforce lower transmit power at borders, but the issue persists,
complicating the user experience for traders and border communities.^15
```
## 6. Impact on Quality of Service (QoS) and Economic

## Life

The convergence of these interference sources—boosters, illegal radio, rain fade, and border


spillover—translates into tangible economic and functional losses.

### 6.1 Measured QoS Degradation

UCC benchmark audits reveal the disparity between voice and data performance.

```
● Voice vs. Data: While voice call setup success rates are generally high (above 95% in
most towns), data metrics suffer. In Kampala's dense urban core (e.g., Arua Park), latency
and packet loss are high. This is consistent with a high-interference environment where
the network is constantly re-transmitting lost packets due to noise.^11
● Blocked Calls: In areas with heavy "booster" usage, blocked call rates can spike above
the 2% regulatory threshold, as the base station simply cannot hear the requests from
mobile phones over the noise.^32
```
### 6.2 The Economic Toll

```
● Operator Costs: MTN and Airtel are forced to invest millions in "spectrum cleaning"
teams—engineers who drive around with spectrum analyzers hunting down illegal
interference sources. This is capital that could otherwise be spent on network expansion.
● The Digital Divide: For the 7.5 million Ugandans currently unconnected, the high cost of
ensuring reliable service in high-interference, difficult-terrain areas acts as a deterrent
for operators to expand coverage, exacerbating the digital divide.^4
```
## 7. Mitigation Strategies and Strategic Outlook

Addressing these challenges requires a synchronized approach involving strict enforcement,
advanced engineering, and regional diplomacy.

### 7.1 Regulatory Force: The Clean Spectrum Initiative

The UCC has adopted a zero-tolerance posture towards spectrum pollution.

```
● Enforcement Operations: The commission actively confiscates illegal equipment. In
2024 alone, operations in Masaka and Mubende targeted illegal radio stations and
megaphones, shutting them down to protect the noise floor.^21
● Type Approval: Strengthening the border controls to prevent the importation of
non-compliant telecommunications equipment is a primary preventative measure.
```
### 7.2 Technological Resilience

Operators are deploying "interference-aware" technologies.

```
● Massive MIMO & Beamforming: 5G technology utilizes Massive MIMO (Multiple Input
Multiple Output). Unlike older antennas that broadcast in a wide sector, Massive MIMO
can form narrow beams of energy directed at a specific user. This spatial filtering allows
the antenna to "ignore" interference coming from other directions, significantly improving
```

```
performance in noisy urban environments.^33
● Fiber-to-the-Tower (FTTT): The ultimate solution to rain fade is removing the air gap in
the backhaul. The expansion of the NBI allows operators to connect more towers directly
to fiber, rendering them immune to atmospheric attenuation.^14
● Adaptive Power Control: Modern networks dynamically adjust the power of the mobile
device and the tower thousands of times per second. By keeping transmit power to the
absolute minimum required, the overall noise pollution in the cell is reduced.^34
```
### 7.3 Regional Harmonization

```
● Spectrum Coordination: Through the East African Community (EAC), Uganda is
harmonizing its spectrum roadmap with neighbors. This involves agreeing on "guard
bands" and specific frequency blocks for border areas to prevent overlap.^15
```
## 8. Conclusion

Uganda's mobile network landscape is a testament to rapid technological leapfrogging, where
5G networks are rising in the cities even as 2G remains essential in the villages. Coverage is no
longer the primary constraint; the challenge has shifted to **quality** and **capacity**. The integrity
of the network is under constant siege from a chaotic electromagnetic environment—fueled
by illegal boosters, unlicensed broadcasters, and the fierce equatorial weather.

The path forward lies in the "sanitization" of the spectrum. Without aggressive policing of
illegal interference sources and continued investment in weather-resilient fiber backhaul, the
theoretical speeds of 5G will remain unattainable for the average user. The synergy between
the UCC’s enforcement arm and the operators' technical innovation will define the next
decade of Uganda’s digital trajectory. As the National Backbone Infrastructure closes the
physical gaps, the industry must now close the "signal gap" created by noise, ensuring that
the 96% coverage statistic translates into 100% meaningful connectivity.

#### Works cited

#### 1. Market Performance Report – THE COMMUNICATIONS BLOG, accessed on

#### January 8, 2026, https://uccinfoblog.com/tag/market-performance-report/

#### 2. It's time to utilize Uganda's impressive internet broadband coverage - The

#### Observer, accessed on January 8, 2026,

#### https://observer.ug/viewpoint/its-time-to-utilize-ugandas-impressive-internet-br

#### oadband-coverage/

#### 3. Spectrum assignment moves Uganda closer to national broadband targets -

#### GSMA, accessed on January 8, 2026,

#### https://www.gsma.com/connectivity-for-good/spectrum/spectrum-assignment-

#### moves-uganda-closer-to-national-broadband-targets/

#### 4. 7.5 Million Ugandans Cut Off as Mobile Access Gap Widens – UCC - Parliament

#### Watch, accessed on January 8, 2026,


#### https://parliamentwatch.ug/news-amp-updates/7-5-million-ugandans-cut-off-as

#### -mobile-access-gap-widens-ucc/

#### 5. Airtel Uganda expands 5G footprint to 365 sites ahead of festive season - UG

#### Bulletin, accessed on January 8, 2026,

#### https://www.ugbulletin.co.ug/airtel-uganda-expands-5g-footprint-to-365-sites-a

#### head-of-festive-season/

#### 6. Airtel Uganda Hits 365 5G Sites Across Regions as Holiday Season Kicks in - Nile

#### Post, accessed on January 8, 2026,

#### https://nilepost.co.ug/business/310029/airtel-uganda-hits-365-5g-sites-across-re

#### gions-as-holiday-season-kicks-in

#### 7. From connectivity to community: MTN Uganda unveils inaugural 2024

#### sustainability report highlighting growth, green initiatives & social impact - Eagle

#### Online, accessed on January 8, 2026,

#### https://eagle.co.ug/2025/08/26/from-connectivity-to-community-mtn-uganda-un

#### veils-inaugural-2024-sustainability-report-highlighting-growth-green-initiatives-

#### social-impact/

#### 8. MTN WakaNet, accessed on January 8, 2026,

#### https://www.mtn.co.ug/personal/mtn-wakanet/

#### 9. MTN Mobile 3G / 4G / 5G coverage in Kampala, Central Region, Uganda -

#### nPerf.com, accessed on January 8, 2026,

#### https://www.nperf.com/en/map/UG/232422.Kampala/223581.MTN-Mobile/signal

#### 10. Airtel's 3G / 4G / 5G coverage map - Wakiso, Uganda - nPerf.com, accessed on

#### January 8, 2026,

#### https://www.nperf.com/en/map/UG/225964.Wakiso/1639.Airtel/signal

#### 11. quality of service findings for indoor mobile voice telephony and data services in

#### uganda - AWS, accessed on January 8, 2026,

#### https://newvision-media.s3.amazonaws.com/cms/b295e426-f529-44b6-928d-

#### 330d8b911.pdf

#### 12. QOS August to September 2024, accessed on January 8, 2026,

#### https://www.ucc.co.ug/wp-content/uploads/2024/12/QOS-August-to-September

#### -2024.pdf

#### 13. MTN Mobile's 3G / 4G / 5G coverage map in Uganda - nPerf.com, accessed on

#### January 8, 2026, https://www.nperf.com/en/map/UG/-/223581.MTN-Mobile/signal

#### 14. President Yoweri K Museveni launches NBI PHASE V in Karamoja - NITA-U,

#### accessed on January 8, 2026,

#### https://www.nita.go.ug/nita/news-and-updates/president-yoweri-k-museveni-lau

#### nches-nbi-phase-v-karamoja

#### 15. EAC Partner States Push to Cut Roaming Costs and Boost Cross-Border

#### Connectivity, accessed on January 8, 2026,

#### https://www.eac.int/eadrip-news-updates/eardip-press-releases/3425-eac-partn

#### er-states-push-to-cut-roaming-costs-and-boost-cross-border-connectivity

#### 16. NITA-U, OFFICE OF THE PRESIDENT, MINISTRY OF ICT AND NATIONAL GUIDANCE

#### COLLABORATE TO EXTEND NBI PHASE 5 TO KANUNGU DISTRICT, accessed on

#### January 8, 2026,

#### https://www.nita.go.ug/nita/news-and-updates/nita-u-office-president-ministry-i


#### ct-and-national-guidance-collaborate-extend

#### 17. Gov't to extend high-speed internet to all sub-counties | Parliament of Uganda,

#### accessed on January 8, 2026,

#### https://www.parliament.go.ug/news/3448/govt-extend-high-speed-internet-all-s

#### ub-counties

#### 18. UCC warns public against use of mobile phone signal boosters - New Vision,

#### accessed on January 8, 2026,

#### https://www.newvision.co.ug/category/news/ucc-warns-public-against-use-of-m

#### obile-phone-NV_

#### 19. MPs decry exploitation of Ugandans by telecom companies - The Independent

#### Uganda, accessed on January 8, 2026,

#### https://www.independent.co.ug/mps-decry-exploitation-of-ugandans-by-teleco

#### m-companies/

#### 20. PUBLIC NOTICE: SIGNAL INTERFERENCE ARISING OUT OF USAGE OF NETWORK

#### REPEATERS – “BOOSTERS” - the communications blog, accessed on January 8,

#### 2026,

#### https://uccinfoblog.com/2021/07/26/public-notice-signal-interference-arising-out

#### -of-usage-of-network-repeaters-boosters/

#### 21. UCC OPERATION TARGETS ILLEGAL RADIO STATIONS, BOOSTERS AND

#### MEGAPHONES - the communications blog, accessed on January 8, 2026,

#### https://uccinfoblog.com/2023/07/05/ucc-operation-targets-illegal-radio-stations-

#### boosters-and-megaphones/

#### 22. UCC, POLICE LAUNCH OPERATION AGAINST ILLEGAL BROADCASTERS,

#### accessed on January 8, 2026,

#### https://uccinfoblog.com/2024/10/22/ucc-police-launch-operation-against-illegal-

#### broadcasters/

#### 23. UCC cracks down on illegal and non-compliant broadcasters, accessed on

#### January 8, 2026,

#### https://www.ucc.co.ug/ucc-cracks-down-on-illegal-and-non-compliant-broadca

#### sters/

#### 24. Analysis of interference signal from LTE phone on sound systems - IEEE Xplore,

#### accessed on January 8, 2026, https://ieeexplore.ieee.org/document/6675490/

#### 25. Analysis of Interfered Noise for Sound Systems over LTE Mobile Phones - UPV,

#### accessed on January 8, 2026,

#### https://personales.upv.es/thinkmind/dl/conferences/icwmc/icwmc_2012/icwmc_

#### 012_12_40_20206.pdf

#### 26. Analysis of Rain Attenuation Effects on The Communication System Quality of The

#### Merah Putih Satellite VSAT IP Services Using C-Band and - Komdigi, accessed on

#### January 8, 2026,

#### https://bpostel.komdigi.go.id/index.php/bpostel/article/view/409/

#### 27. Impact of Rain Attenuation on Path Loss and Link Budget in 5G mmWave Wireless

#### Propagation Under South Africa's Subtropical Climate - MDPI, accessed on

#### January 8, 2026, https://www.mdpi.com/2673-4001/6/3/

#### 28. Effects of rain on microwave and satellite communications in equatorial and

#### tropical regions, accessed on January 8, 2026,


#### https://www.researchgate.net/publication/272579342_Effects_of_rain_on_microw

#### ave_and_satellite_communications_in_equatorial_and_tropical_regions

#### 29. Rain Attenuation in 5G Wireless Broadband Backhaul Link and Develop (IoT)

#### Rainfall Monitoring System - The Science and Information (SAI) Organization,

#### accessed on January 8, 2026,

#### https://thesai.org/Downloads/Volume12No5/Paper_1-Rain_Attenuation_in_5G_Wir

#### eless_Broadband.pdf

#### 30. End cross-border network interference - Bahati | Monitor, accessed on January 8,

#### 2026,

#### https://www.monitor.co.ug/uganda/news/national/end-cross-border-network-inte

#### rference-bahati-

#### 31. Cross border frequency coordination - ITU, accessed on January 8, 2026,

#### https://www.itu.int/en/ITU-D/Regional-Presence/Africa/Documents/National%20S

#### pectrum%20Management%20Assistance%20Workshop/CROSS%20BORDER%

#### 0FREQUENCY%20COORDINATION.pdf

#### 32. Full Pages 2024.indd - Uganda Communications Commission, accessed on

#### January 8, 2026,

#### https://www.ucc.co.ug/wp-content/uploads/2024/02/QoS-FINDINGS-FOR-MOBIL

#### E-VOICE-TELEPHONY-DATA-11-22-NOV.-2023-23.01.2024.pdf

#### 33. Overview on Technologies for Combating Interference and Noise Management in

#### 5G and Beyond Network - Everant Journals, accessed on January 8, 2026,

#### https://everant.org/index.php/etj/article/download/1414/1029/

#### 34. A Review on Rain Signal Attenuation Modeling, Analysis and Validation

#### Techniques: Advances, Challenges and Future Direction - MDPI, accessed on

#### January 8, 2026, https://www.mdpi.com/2071-1050/14/18/



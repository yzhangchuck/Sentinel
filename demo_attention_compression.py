#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Attention-Based Text Compression

This script demonstrates the clean AttentionCompressor implementation with
various configuration options for attention-based text compression.

Usage:
    python demo_attention_compression.py

Setup:
    1. Update DETECTOR_BASE_PATH to point to your detector models directory
    2. Place detector models in the specified directory with the configured names
    3. Ensure all required dependencies are installed (see requirements)

Models Directory Structure:
    models/
    └── detectors/
        ├── qwen2.5-0.5b-instruct-3000_all_layer_last_token_qwen2_20250507_212739_model.pkl
        └── qwen2.5-0.5b-instruct-3000_all_layer_last_token_20250515_033938_model.pkl
"""

import json
import time
import os
from attention_compressor import AttentionCompressor

# Configuration - Update these paths according to your setup
DETECTOR_BASE_PATH = os.path.join(os.path.dirname(__file__), "models", "detectors")
DETECTOR_CONFIGS = {
    "standard": "qwen2.5-0.5b-instruct-3000_all_layer_last_token_qwen2_20250507_212739_model.pkl",
    "selected_idx": "qwen2.5-0.5b-instruct-3000_all_layer_last_token_20250515_033938_model.pkl"
}

chinese_question = "姚明受过几次伤"
chinese_context ="""姚明（1980年9月12日—），中国篮球运动员，生于上海市徐汇区，祖籍江苏吴江，曾为中国国家篮球队队员，亦曾效力于中国篮球职业联赛（CBA）上海大鲨鱼篮球俱乐部和美国国家篮球协会（NBA）休斯敦火箭，外号“小巨人”“移动长城”（The Walking Great Wall）。退役后一度出任中国篮球协会主席。
1998年4月，姚明入选王非执教的国家队，开始了职业篮球生涯。并在中国篮球协会（CBA）的上海大鲨鱼效力了五年。2001夺得CBA常规赛最有价值球员及联赛最有价值球员[2]，2002年获得了CBA总冠军，但该年MVP由刘玉栋获得。[3]分别三次当选CBA篮板王以及CBA盖帽王，二次当选CBA扣篮王。
姚明是中国最具影响力的人物之一，同时也是世界最知名的华人运动员之一[4][5]。2009年，姚明收购上海男篮，成为上海大鲨鱼篮球俱乐部老板[6]。2011年7月20日，姚明正式宣布退役[7][8]。2016年11月22日，姚明出任CBA联盟副董事长。2017年2月，姚明当选为中国篮球协会主席[9]。2016年4月4日，姚明与前NBA球星沙奎尔·奥尼尔和艾伦·艾弗森一同入选奈史密斯篮球名人纪念堂，他也是首位入选也是迄今为止唯一入选名人堂的亚洲球员。2024年10月31日，姚明辞去中国篮球协会主席[10]。
青年时代及CBA生涯[编辑]
1980年9月12日傍晚，姚明生于上海徐汇区上海市第六人民医院，他是家中的独子，父亲姚志源身高6英尺10英寸（2.08米），母亲方凤娣身高6英尺2英寸（1.88米）[11]。两人之前都是职业篮球运动员[12]。姚明出生时重11磅（5.0千克）[13]，身高大约是中国新生儿平均身高的两倍。[14]姚明10岁时，他的身高已达5英尺5英寸（1.65米）。当时，一位给姚明做了身体检查的体育医生曾预言，姚明将来会长到7英尺3英寸（2.20米）[14]。姚明9岁在上海高安路第一小学就读时开始打篮球，后进入少年体校[15]。
姚明13岁时首次登场，当时他效力于中国篮球联赛（CBA）中的上海大鲨鱼少年队。姚明为了入选球队，每天训练10小时。[16]在少年队打了4年球后，17岁的姚明进入了大鲨鱼队，并在新秀赛季中平均每场得到10分和8个篮板球。然而，在接下来的一个赛季中姚明因为职业生涯的第二次脚伤不得不告别下面的比赛。当时他自称弹跳力仅为3至6英寸（10至15cm）[17]。姚明在CBA的1999-2001两个赛季中，大鲨鱼队都闯入了CBA的决赛，但在这两场比赛中，大鲨鱼都输给了八一火箭队。在接下来的2001-2002赛季中，王治郅离开八一火箭队并成为第一位挑战NBA的中国球员，这使得大鲨鱼队终于摘取了2001-2002赛季CBA的总冠军。在上海队的最后一个季后赛中，姚明平均每场获得38.9分和20.2个篮板，同时，他的远距离射篮的命中率为76.7%[18]，并在决赛的关键一战中创下了21投21中的记录[19]。
姚明在1997-98赛季平均每场获10分和9个篮板球，但是由于伤病错过了下个赛季的大部分比赛。在1999-2000赛季中，康复归队的他每场得分21分，抢14个篮板球，及5次助攻。2000-01赛季是他CBA生涯最辉煌的时期，20岁的姚明凭借每场得分27分和19个篮板被评为常规赛最有价值球员（MVP）及联赛最有价值球员。[2]

NBA职业生涯[编辑]参加NBA选秀[编辑]
1999年，上海大鲨鱼队副总经理李耀明催促姚明参加1999年NBA选秀[13]。李耀明还促成姚明与美国常青体育公司（Evergreen Sports Inc.）签署合约，由该公司作姚明的代理机构。该合约规定常青公司将获得姚明收入的三分之一[13]，但后来该合同被确定无效[20]。
当姚明决定参加2002年NBA选秀时，一个集合经纪人和智囊团的顾问团队开始成形，后来人们称之为“姚之队”。这个团队包括姚明的代理人章明基（英文名：Erik Zhang），NBA经纪人比尔·达菲（Bill Duffy），中方经纪人陆浩，芝加哥大学经济学教授约翰·赫伊津哈（John Huizinga），[21]以及BDA体育管理的市场副总裁比尔·桑德斯（Bill Sanders）。[22]人们普遍预测姚明将在选秀中拔得头筹。[23][24][25]然而有些球队担心姚明能否进入NBA，因为当时CBA是否允许姚明在美国打球尚不确定。[26]
姚明在2002年的NBA新秀选拔赛被休斯敦火箭队以第一顺位选中。火箭队的明星中锋阿基姆·奥拉朱万在一年前刚刚离队，他们相信姚明可以填补这个空白。由于王治郅与国家队出赛产生误会并导致其被开除出国家队，[27]CBA合约规定姚明必须回国代表国家队出赛。[28]他们同时表示，除非休斯敦火箭队愿意在第一轮选秀中选择姚明，否则他们将不允许他到美国打球。[29]在“姚之队”保证火箭队将把头号选秀权用于姚明后，CBA于选秀日当天早晨批准姚明到美国打球。[30]当火箭队以第一顺位选中姚明后，他成为历史上第一个没有美国高校篮球经验，却当选状元秀的国际球员。[31]
在2002年季后赛中，姚明随中国国家篮球队参加了在美国印第安纳波利斯举行的世界篮球锦标赛，因而错过火箭队季前集训[32]。赛后，姚明与NBA球星马努·吉诺比利、德克·诺维茨基及佩贾·斯托贾科维奇切磋了球技。
2002－03赛季[编辑]
姚明没有参加火箭队的集训，而是代表中国参加2002年国际篮联（FIBA）举办的世界篮球锦标赛。[33]赛季开始前，包括比尔·西蒙斯（Bill Simmons）和迪克·维塔勒（Dick Vitale）在内的数位篮球评论员预言姚明将败走NBA。[34][35][36]评论员查尔斯·巴克利打赌说如果姚明在新秀赛季中单场得分超过19分，他就“亲吻肯尼·史密斯（Kenny Smith）的臀部”。[37]姚明的第一场NBA比赛对阵的是印第安纳步行者队，取得两个篮板无得分，[38][39]而后在与丹佛掘金的比赛中首开纪录。[40]在他最初的七场比赛中，他平均上场仅14分钟，场均得4分。但是在11月17日对阵湖人的比赛中，姚明表现完美，9投9中加上罚球2投2中，共得到20分。[41]巴克利遵照赌约亲吻了史密斯带来的驴的臀部（ass有臀部和驴两种意思）。[37]
姚明与沙奎尔·奥尼尔（Shaquille O'Neal ）的第一次对话是在2003年1月17日，在此之前，奥尼尔曾在媒体前说“告诉姚明，Ching chong-yang-wah-ah-soh”，这句话立即引起亚裔美籍人士反感，认为有种族歧视之嫌[42]。奥尼尔之后否认这一说法，并表示这不过是句玩笑话。[43]姚明也表示相信奥尼尔当时是在开玩笑，[43]但是媒体对二人言论的关注不断上升，进而增加了这场全国电视直播比赛的火药味。姚明在比赛一开场拿下6分并送奥尼尔两记盖帽，在加时赛最后10秒又大力扣篮，锁定胜局。[44]姚明全场拿下10分10个篮板；奥尼尔则攻下31分13个篮板。[45]
姚明以每场平均13.5的得分，8.2个篮板球的成绩结束了他的新人赛季，[46]获得NBA年度新秀奖第二名（阿玛雷·斯塔德迈尔为当年最佳新秀），[47]赢得一致推选进入NBA年度最佳新人阵容，[48]当选美国《体育新闻》杂志年度最佳新人奖，[49]并获得劳伦斯年度最佳新秀奖。 [50]
2月，姚明被选为2003年NBA全明星赛西部联队首发中锋。尽管在短短的16分钟的出场时间里，他只得到了2分和2个篮板球，但他的当选证明他在球迷心目中的的受欢迎程度，他的选票数甚至比超级中锋奥尼尔的还要多。许多球迷对姚明的此次当选表示质疑，怀疑如果没有来自中国的数量庞大的球迷的投票，他不可能以一个新秀的身份当选全明星的首发中锋，因为NBA在2002-03赛季首次开通网上投票产生全明星球员。这些球迷批评说中国庞大的网友数量影响了投票的公平性。姚明在西部联队中锋位置的竞争对手沙奎尔·奥尼尔也指责说姚明之所以能当选主要是因为众多中国球迷的支持。虽然有很多质疑的声音，但是并没有证据证明姚明的此次当选中国球迷的投票是决定性因素（在2004年的全明星赛之前，他的网上得票数比他的竞争对手奥尼尔少了4,000票，在实物投票（纸票）中占尽优势。这些纸制选票在美国的NBA赛场和购物中心派发，中国本土球迷较难获得，实物投票的优势令姚明的总票数多出奥尼尔29,000票）。
在2003年全明星赛之后，姚明发挥稳定。整个赛季结束后，他的平均得分为13.5分，平均每场8.2个篮板球[51]，新秀中排名第二；平均每场1.8次封盖，新秀中排名第一；新秀综合排名第二位，仅次于菲尼克斯太阳队的阿玛雷·斯塔德迈尔[52]在2003年NBA的季后赛中，姚明竭尽全力筹集资金抵抗非典型性肺炎（SARS）。通过一部系列电视片，他筹得30万美元用以帮助抵抗疾病。
2003－04赛季[编辑]
在姚明的二年级赛季开始前，火箭队主教练鲁迪·汤姆贾诺维奇因身体原因辞职，[53]曾长期在纽约尼克斯队担任主教练的杰夫·范甘迪（Jeff Van Gundy）接班上任。范甘迪注重培养姚明的进攻能力，[54] 这使姚明在该赛季平均得分和篮板上均再创新高，2004年2月在击败亚特兰大老鹰队的三个加时战中砍下41分7个助攻，创职业生涯最高分纪录。[55]在2004年的全明星赛中，姚明继2003年之后再次当选西部联队首发中锋，[56]以平均每场17.5分，9.0个篮板球结束了他的第二个赛季，[46] 更是在自己的职业生涯当中，首次帮助火箭队在2003-04赛季以西部联盟第7名的成绩杀入季后赛。不过可惜的是在第一轮与湖人队进行5场比赛后就被淘汰了。[57]姚明首次季后赛平均每场15分7.4个篮板。[46] 在2004年的雅典奥运会的开幕式上，姚明高举中国国旗率代表队入场。随后他爆炸性地宣布：如果中国国家篮球队打不进四分之一决赛，他将半年不刮胡子。经过几场低迷比赛（中国队以58－83、57－82和52－89先后负于西班牙、阿根廷和意大利）后，中国队奇迹般地以67-66的微弱优势战胜了世界冠军强队——塞尔维亚和黑山队，姚明一人独得27分。由于他的出色表现：平均每场20.7分、9.3个篮板球和高达55.9%的投篮命中率，他入选了奥运会全明星阵容。
2004－05赛季[编辑]
2004年夏天，在与奥兰多魔术队的一次七人大交易中，火箭队签入特雷西·麦克格雷迪，同时送出史蒂夫·弗朗西斯和卡蒂诺·莫布里。[58]尽管姚明表示弗兰西斯和莫布里“在（他）前两个赛季对（他）帮助很大”，但他补充道，“我很兴奋能同特雷西·麦克格雷迪并肩作战。他能带来惊喜。”[59]外界预测，这样的一支火箭队将成为夺标大热门。[58][60] 2005年，姚麦二人入选全明星赛首发阵容。姚明获得2,558,278张选票，打破了先前由迈克尔·乔丹保持的全明星最高得票数纪录，这是由于他的劲敌沙奎尔·奥尼尔在季中转到东部的迈阿密热火队中，但这同时也暗示着姚明已经步入NBA超级巨星的行列。[61]。2005年3月11日，在与菲尼克斯太阳队的比赛中，姚明获得27分，抢下22个篮板球，并有5次封盖，拿到最高的“两双”战绩。在2005年NBA季后赛中，火箭队以51胜，西部排名第五，连续第二年冲进季后赛。季后赛遭遇达拉斯独行侠队，[62]火箭队客场先胜两场，姚明第二场14投13中，创下火箭队史上季后赛最高投篮命中率。[63]然而接下来4场火箭队不敌独行侠队，第七场更是落后40分，这是NBA历史上季后赛第七场的最大分差。[64]，姚明平均每场得分21.4分、7.7个篮板球和2.71次封盖，并且有三场得分在30分以上。
2005年，《姚明年》发行，影片主要记录了他来美国第一年所发生的事。
姚明在他的前三个赛季里向观众充分证明了自己的能力。在他的前两个赛季中，他没有缺席任何一场比赛；在第三赛季仅缺席2场。但是在第四赛季中，姚明被列入伤病休息名单中，原因是他的左脚拇趾患严重的甲沟炎[65]。这个病症自从在季前赛的一次比赛中，他的左脚拇趾趾甲脱落后就一直困扰着他，姚明抱怨说那是在客场与西雅图超音速队的比赛中，正是丹尼·福特森导致了他的脚伤。2005年12月18日，当队友们在洛杉矶与湖人队客场厮杀时，姚明返回休斯敦进行脚趾手术。随后他被列入休息名单中，估计6至8周不能比赛。在错过了21场比赛之后，姚明重新回到了赛场，并在与孟菲斯灰熊队的比赛中得15分。
2006－07赛季[编辑]
这个赛季姚明场均得到25分、9.4个篮板和2个封盖，得分首次超过麦迪。但由于在与洛杉矶快船队的比赛中胫骨受伤，姚明这个赛季常规赛出场仅为48场。在季后赛中，他场均贡献25.1分、10.3篮板，不过火箭队还是以3比4不敌犹他爵士队止步季后赛第一轮。
姚明在2006年全明星赛以1,319,868票高居榜首，科比·布莱恩特以1,213,387票位居第二。
2007－08赛季[编辑]
2008年2月26日，在率领休斯敦火箭队获得12连胜之后，被诊断舟骨裂缝，手术后需要休息4个月，07-08赛季剩余将无法出场，随后火箭队在麦迪带领下，继续赢下10连胜，创造NBA历史最长连胜纪录的第四位。[66]。姚明在该球季出赛55场，平均取得22分，10个篮板及2个封盖[51]。
2008－09赛季[编辑]
这个赛季姚明场均贡献19.7分、9.9篮板和1.9封盖。另外，他的赛季总得分达到了职业生涯最高的1,514分。他顺利带领火箭进入季后赛，并首轮淘汰波特兰开拓者队，火箭队在时隔多年后再次进入季后赛第二轮。值得一提的是，姚明在首轮对阵开拓者队的第一场中，上半场24分钟内，9投9中得到24分和9个篮板。不过在第二轮对战洛杉矶湖人队的比赛中，姚明因脚伤导致休战，甚至影响了2009-10赛季。
2009－10赛季[编辑]
由于受到上赛季的伤势影响而动手术，致使该季姚明没有任何一次出场，只能穿着西装观战。
2010－11赛季[编辑]
2010年11月11日，伤愈复出不久的姚明在同奇才队的比赛中再度受伤，随后一直休战。12月18日，火箭官方宣布姚明左脚踝第三次应力性骨折，赛季报销。
退役[编辑]
2011年7月20日，姚明在上海名为“明谢”的新闻发布会上宣布自己将正式退役[67][68]。早在美国东部时间7月8日就有媒体曝出姚明将会退役的消息[69]。直至7月20日，姚明在上海一家酒店新闻发布厅中做了简短的声明，宣布将结束自己的运动生涯[68]。在声明中他提到了自己左脚的第三次应力性骨折，回忆了自己的篮球生涯并发表感谢，最后用英语特别向休斯顿火箭球迷等致谢[67]。国家篮球协会总裁大卫·斯特恩发表声明回应称姚明“在赛场上的统帅力，谦逊的态度，对慈善事业的不懈努力以及他风趣幽默的个性，使他成为在全球范围内备受爱戴的球员。他为中美两国的球迷架起了一座非凡的桥梁。”[70]
退役后[编辑]
退役后的姚明依然活跃在公众视野内，作为上海大鲨鱼篮球俱乐部老板，以自己的影响力为中国篮球做出贡献。他先后当选上海公共外交协会副会长兼荣誉大使、上海市第十一届政协委员、上海市第十一届政协委员会常务委员，2013年当选为第十二届全国政协委员[71]。
2016年2月，由18支中国男子篮球职业联赛参赛球队共同出资建立的“中职联篮球俱乐部（北京）股份有限公司”在北京成立。姚明担任公司董事长，并暂时兼任公司总经理。[72]公司计划将深度参与CBA职业联赛的改革。但是4月时，姚明确认和中国篮协的第二次沟通协商结束，双方在赛事改革上没有达成任何共识。[73]姚明退役后进入上海交通大学深造，就读经济学类本科专业[1]。2018年7月9日，于上海交通大学毕业。[9]
美国时间2017年2月3日，在休斯顿火箭主场对芝加哥公牛中场休息期间宣布将其效力火箭的11号球衣高挂退役，成为NBA首位球衣退役的中国球员。[74]
2017年2月23日，在中国篮球协会第九届全国代表大会中，姚明当选主席。[9]2024年10月辞任。[75]
家庭[编辑]
其妻叶莉，前中国女篮队员，姚明17岁就与她相识。2007年8月3日，姚明和叶莉在上海徐汇区婚姻登记处正式领取结婚证书，2007年8月6日，两人举办婚礼[76]。
其女姚沁蕾（Amy），2010年5月22日在美国休斯敦出生[77]，取名Amy[78][79][80]，拥有美国国籍[81]，在美国长大，为美国公民。姚明曾表示等他女儿成年，再自行决定是否要申请中华人民共和国国籍，放弃美国国籍[82]。
"""


english_question = "How many times did Yao Ming get injured?"
english_context = """Yao Ming (Chinese: 姚明; born September 12, 1980) is a Chinese basketball executive and former professional player. He played for the Shanghai Sharks of the Chinese Basketball Association (CBA) and the Houston Rockets of the National Basketball Association (NBA). Yao was selected to start for the Western Conference in the NBA All-Star Game eight times, and was named to the All-NBA Team five times. During his final season, he was the tallest active player in the NBA, at 7 feet 6 inches (2.29 m).[1]
Yao, who was born in Shanghai, started playing for the Sharks as a teenager, and played on their senior team for five years in the CBA, winning a championship in his final year. After negotiating with the CBA and the Sharks to secure his release, Yao was selected by the Rockets as the first overall pick in the 2002 NBA draft. He reached the NBA playoffs four times, and the Rockets won the first-round series in the 2009 postseason, their first playoff series victory since 1997. In July 2011, Yao announced his retirement from professional basketball because of a series of foot and ankle injuries which forced him to miss 250 games in his last six seasons.[2] In eight seasons with the Rockets, Yao ranks sixth among franchise leaders in total points and total rebounds, and second in total blocks.[3]
Yao is one of China's best-known athletes internationally, with sponsorships with several major companies. His rookie year in the NBA was the subject of a documentary film, The Year of the Yao, and he co-wrote, along with NBA analyst Ric Bucher, an autobiography titled Yao: A Life in Two Worlds. Known in China as the "Yao Ming Phenomenon" and in the United States as the "Ming Dynasty", Yao's success in the NBA, and his popularity among fans, made him a symbol of a new China that was both more modern and more confident.[4] Yao is also an entrepreneur and owner of Yao Family Wines in Napa Valley, California.[5]
In April 2016, Yao was elected into the Naismith Memorial Basketball Hall of Fame, alongside Shaquille O'Neal and Allen Iverson, becoming the first Chinese national to be inducted into the Hall of Fame.[6][7] In February 2017, Yao was unanimously elected as chairman of the Chinese Basketball Association.[8] Yao had a storied career as a member of the Chinese national team.[9] With the national team, Yao won the FIBA Asia Cup in 2001, 2003, and 2005, winning MVP of the tournament all three times.[10] He also made the All-Tournament Team at the FIBA World Cup in 2002. Yao retired from the Chinese national team after the 2008 Beijing Olympics.[11][12]
Early life
Yao Ming was born on September 12, 1980, in Shanghai, China.[13] He is the only child of 6-foot-7-inch (2.01 m) Yao Zhiyuan and 6-foot-3-inch (1.91 m) Fang Fengdi,[14] both of whom were former professional basketball players.[15] At 11 pounds (5.0 kg), Yao weighed more than twice as much as the average Chinese newborn.[16] When Yao was nine years old, he began playing basketball and attended a junior sports school.[17] The following year, Yao measured 5 feet 5 inches (1.65 m)[18] and was examined by sports doctors, who predicted he would grow to 7 feet 3 inches (2.21 m).[18]
Professional careerShanghai Sharks (1997–2002)
Yao first tried out for the Shanghai Sharks' junior team of the Chinese Basketball Association (CBA) when he was 13 years old, and practiced ten hours a day for his acceptance.[19] After playing with the junior team for four years, Yao joined the Sharks' senior team, where he averaged 10 points and 8 rebounds a game in his rookie season. His next season was cut short when he broke his foot for the second time in his career, which Yao said decreased his jumping ability by four to six inches (10 to 15 cm).[20] The Sharks made the finals of the CBA in Yao's third season and again the next year, but lost both times to the Bayi Rockets. When Wang Zhizhi left the Bayi Rockets to become the first NBA player from China the following year, the Sharks finally won their first CBA championship. During the playoffs in his final year with Shanghai, Yao averaged 38.9 points and 20.2 rebounds a game, while shooting 76.6% from the field,[21] and made all 21 of his shots during one game in the finals.[22]
Houston Rockets (2002–2011)
Yao was pressured to enter the NBA draft in 1999 by Li Yaomin, the deputy general manager of the Shanghai Sharks.[16] Li also influenced Yao to sign a contract for Evergreen Sports Inc. to serve as his agent. The agreement entitled Evergreen to 33% of Yao's earnings,[16] but the contract was later determined to be invalid.[23]
As American attention on Yao grew, Chinese authorities also took interest. In 2002, the Chinese government released new regulations that would require him and other Chinese players to turn over half of any NBA earnings to the government and China's national basketball association, including endorsements as well as salaries.[24]
When Yao decided to enter the 2002 NBA draft, a group of advisers was formed that came to be known as "Team Yao". The team consisted of Yao's negotiator, Erik Zhang; his NBA agent, Bill Duffy; his Chinese agent, Lu Hao; University of Chicago economics professor John Huizinga;[25] and the vice president for marketing at BDA Sports Management, Bill Sanders.[26] Yao was widely predicted to be picked number one overall.[27][28][29] However, some teams were concerned about Yao's NBA eligibility because of uncertainty over whether the CBA would let Yao play in the United States.[30]
Shortly after Wang Zhizhi refused to return to China to play for the national team and was subsequently banned from playing for China,[31] the CBA stipulated that Yao would have to return to play for the national team.[32] They also said they would not let him go to the United States unless the Houston Rockets would take him first overall.[33] After assurances from Team Yao that the Rockets would draft Yao with their number one pick, the CBA gave permission on the morning of the draft for Yao to play in the U.S.[34] When the Rockets selected Yao with the first pick of the draft, he became the first international player ever to be selected first overall without having previously played U.S. college basketball.[35]
Beginning years (2002–2005)
Yao did not participate in the Rockets' pre-season training camp, instead playing for China in the 2002 FIBA World Championships.[36] Before the season, several commentators, including Bill Simmons and Dick Vitale, predicted that Yao would fail in the NBA,[37][38] and Charles Barkley said he would "kiss Kenny Smith's ass" if Yao scored more than 19 points in one of his rookie-season games.[39] Yao played his first NBA game against the Indiana Pacers, scoring no points and grabbing two rebounds,[40][41] and scored his first NBA basket against the Denver Nuggets.[42] In his first seven games, he averaged only 14 minutes and 4 points, but on November 17, he scored 20 points on a perfect 9-of-9 from the field and 2-of-2 from the free-throw line against the Lakers.[43] Barkley made good on his bet by kissing the buttock of a donkey purchased by Smith for the occasion (Smith's "ass").[39]
In Yao's first game in Miami on December 16, 2002, the Heat passed out 8,000 fortune cookies, an East Asian cultural stereotype.[44][45] Yao was not angry with the promotion because he was not familiar with American stereotypes of Chinese.[46] In an earlier interview in 2000, Yao said he had never seen a fortune cookie in China and guessed it must have been an American invention.[47]
Before Yao's first meeting with Shaquille O'Neal on January 17, 2003, O'Neal said, "Tell Yao Ming, ching chong-yang-wah-ah-soh", prompting accusations of racism.[46] O'Neal denied that his comments were racist, and said he was only joking.[48] Yao also said he believed O'Neal was joking, but he said a lot of Asians would not see the humor.[48][49] In the game, Yao scored the Rockets' first six points of the game and blocked O'Neal twice in the opening minutes as well as altering two other shots by O'Neal, all 4 of those attempts coming right at the rim, and made a game-sealing dunk with 10 seconds left in overtime.[50] Yao finished with 10 points, 10 rebounds, and 6 blocks; O'Neal recorded 31 points, 13 rebounds, and 0 blocks.[51] O'Neal later expressed regret for the way he treated Yao early in his career.[52]
The NBA began offering All-Star ballots in three languages—English, Spanish and Chinese—for fan voting of the starters for the 2003 NBA All-Star Game.[53] Yao was voted to start for the West over O'Neal, who was coming off three consecutive NBA Finals MVP Awards.[54] Yao received nearly a quarter million more votes than O'Neal, and he became the first rookie to start in the All-Star Game since Grant Hill in 1995.[55]

Yao prepares to shoot a free throw with John Stockton in the background
Yao finished his rookie season averaging 13.5 points and 8.2 rebounds per game,[56] and was second in the NBA Rookie of the Year Award voting to Amar'e Stoudemire,[57] and a unanimous pick for the NBA All-Rookie First Team selection.[58] He was also voted the Sporting News Rookie of the Year,[59] and won the Laureus Newcomer of the Year award.[60]Before the start of Yao's sophomore season, Rockets' head coach Rudy Tomjanovich resigned because of health issues,[61] and long-time New York Knicks head coach Jeff Van Gundy was brought in. After Van Gundy began focusing the offense on Yao,[62] Yao averaged career highs in points and rebounds for the season, and had a career-high 41 points and 7 assists in a triple-overtime win against the Atlanta Hawks in February 2004.[63] He was also voted to be the starting center for the Western Conference in the 2004 NBA All-Star Game for the second straight year.[64] Yao finished the season averaging 17.5 points and 9.0 rebounds a game.[56] The Rockets made the playoffs for the first time in Yao's career, claiming the seventh seed in the Western Conference. In the first round, however, the Los Angeles Lakers eliminated Houston in five games.[65] Yao averaged 15.0 points and 7.4 rebounds in his first playoff series.[56]
In the summer of 2004, the Rockets acquired Tracy McGrady from the Orlando Magic in a seven-player trade that also sent Steve Francis and Cuttino Mobley to Orlando.[66] Although Yao said that Francis and Mobley had "helped [him] in every way [his] first two seasons", he added, "I'm excited about playing with Tracy McGrady. He can do some amazing things."[67] After the trade, it was predicted that the Rockets would be title contenders.[66][68] Both McGrady and Yao were voted to start in the 2005 NBA All-Star Game, and Yao broke the record previously held by Michael Jordan for most All-Star votes, with 2,558,278 total votes.[69] The Rockets won 51 games and finished fifth in the West, and made the playoffs for the second consecutive year, where they faced the Dallas Mavericks.[70] The Rockets won the first two games in Dallas, and Yao made 13 of 14 shots in the second game, the best shooting performance in the playoffs in Rockets history.[71] However, the Rockets lost four of their last five games and lost Game 7 by 40 points, the largest Game 7 deficit in NBA history.[72] Yao's final averages for the series were 21.4 points on 65% shooting and 7.7 rebounds.[56]
Career highs and injury-plagued seasons (2005–2011)

In his fifth season, Yao averaged a career-high 25 points per game.
After missing only two games out of 246 in his first three years of NBA play,[14] Yao was rewarded with a five-year, $75 million extension during the 2005 offseason.[73] However, he endured an extended period on the inactive list in his fourth season after developing osteomyelitis in the big toe on his left foot, and surgery was performed on the toe on December 18, 2005.[74] Despite missing 21 games while recovering,[14] Yao again had the most fan votes to start the 2006 NBA All-Star Game.[75]
In 25 games after the All-Star break, Yao averaged 25.7 points and 11.6 rebounds per game, while shooting 53.7% from the field and 87.8% at the free-throw line.[76] His final averages in 57 games were 22.3 points and 10.2 rebounds per game.[56] It was the first time that he ended the season with a so-called "20/10" average. However, Tracy McGrady played only 47 games in the season, missing time because of back spasms.[77] Yao and McGrady played only 31 games together,[78] and the Rockets did not make the playoffs, winning only 34 games.[79] With only four games left in the season, Yao suffered another injury in a game against the Utah Jazz on April 10, 2006, which left him with a broken bone in his left foot. The injury required six months of rest.[80]
Early into his fifth season, Yao was injured again, this time breaking his right knee on December 23, 2006, while attempting to block a shot.[81] Up to that point he had been averaging 26.8 points, 9.7 rebounds and 2.3 blocks per game, and had been mentioned as an MVP candidate.[82][83] Yao was unable to play in what would have been his fifth All-Star game;[84] he was medically cleared to play on March 4, 2007, after missing 32 games.[85]
Despite Yao's absence, the Rockets made the playoffs with the home court advantage against the Utah Jazz in the first round.[86] The Rockets won the first two games, but then lost four of five games[87] and were eliminated in Game 7 at home; Yao scored 29 points—15 in the fourth quarter.[88] Although he averaged 25.1 points and 10.3 rebounds for the series, Yao said afterwards "I didn't do my job".[89] At the end of the season, Yao was selected to the All-NBA Second Team for the first time in his career, after being selected to the All-NBA Third Team twice.[90]
On May 18, 2007, only weeks after the Rockets were eliminated from the playoffs, Jeff Van Gundy was dismissed as head coach.[91] Three days later, the Rockets signed former Sacramento Kings coach Rick Adelman,[92] who was thought to focus more on offense than the defensive-minded Van Gundy.[93][94]

Yao advanced to the second round of the playoffs for the only time in his career in 2009.

Yao playing against Gilbert Arenas
On November 9, 2007, Yao played against fellow Chinese NBA and Milwaukee Bucks player Yi Jianlian for the first time. The game, which the Rockets won 104–88, was broadcast on 19 networks in China, and was watched by over 200 million people in China alone, making it one of the most-watched NBA games in history.[95] In the 2008 NBA All-Star Game, Yao was once again voted to start at center for the Western Conference.[96] Before the All-Star weekend, the Rockets had won eight straight games, and after the break, they took their win streak to 12 games. On February 26, 2008, however, it was reported that Yao would miss the rest of the season with a stress fracture in his left foot. He missed the 2008 NBA playoffs, but he did not miss the 2008 Summer Olympics at Beijing, China in August.[97] After Yao's injury, the Rockets stretched their winning streak to 22 games, at the time the second-longest such streak in NBA history.[98] Yao underwent a successful operation on March 3, which placed screws in his foot to strengthen the bone, and recovery time was estimated at four months.[99] Yao's final averages in 55 games were 22.0 points, 10.8 rebounds, and 2.0 blocks a game.[56]
The next season, Yao played 77 games, his first full season since the 2004–05 season, and averaged 19.7 points and 9.9 rebounds, while shooting 54.8% from the field, and a career-high 86.6% from the free throw line.[56] Despite McGrady suffering a season-ending injury in February,[100] the Rockets finished with 53 wins and the fifth seed in the Western Conference.[101] Facing the Portland Trail Blazers in the first round, Yao finished with 24 points on 9-of-9 shooting in the first game, and the Rockets won 108–81, in Portland.[102] The Rockets won all their games in Houston,[103] and advanced to the second round of the playoffs for the first time since 1997, and the first time in Yao's career.[104]
The Rockets faced the Lakers in the second round, and Yao scored 28 points, with 8 points in the final four minutes, to lead the Rockets to a 100–92 win in Los Angeles.[105] However, the Rockets lost their next two games,[106][107] and Yao was diagnosed with a sprained ankle after Game 3.[108] A follow-up test revealed a hairline fracture in his left foot, and he was ruled out for the remainder of the playoffs.[109] In reaction, Yao said the injury, which did not require surgery, was "better than last year".[110] However, follow-up analysis indicated that the injury could be career threatening.[111] The Yao-less Rockets went on to win Game 4 against the Lakers to even the series 2–2.[112] The Rockets eventually lost the series in seven games.
In July 2009, Yao discussed the injury with his doctors, and the Rockets applied for a disabled player exception, an exception to the NBA salary cap which grants the injured player's team money to sign a free agent.[113] The Rockets were granted the exception, and used approximately $5.7 million on free agent Trevor Ariza. After weeks of consulting, it was decided that Yao would undergo surgery in order to repair the broken bone in his left foot.[114] He did not play the entire 2009–10 season.[115]
For the 2010–11 season, the Rockets said they would limit Yao to 24 minutes a game, with no plan to play him on back-to-back nights. Their goal was to keep Yao healthy in the long term.[115] On December 16, 2010, it was announced that Yao had developed a stress fracture in his left ankle, related to an older injury, and would miss the rest of the season.[116] In January 2011, he was voted as the Western Conference starting center for the 2011 All-Star Game for the eighth time in nine seasons. Injured All-Stars are usually required to attend the All-Star functions and to be introduced at the game, but Yao was not in Los Angeles because of his rehabilitation schedule after his surgery.[117] Yao's contract with the Rockets expired at the end of the season, and he became a free agent.[118]
Retirement
On July 20, 2011, Yao announced his retirement from basketball in a press conference in Shanghai.[119][120] He cited injuries to his foot and ankle, including the third fracture to his left foot sustained near the end of 2010.[121] His retirement sparked over 1.2 million comments on the Chinese social-networking site Sina Weibo.[122] Reacting to Yao's retirement, NBA commissioner David Stern said Yao was a "bridge between Chinese and American fans" and that he had "a wonderful mixture of talent, dedication, humanitarian aspirations and a sense of humor."[121] Shaquille O'Neal said Yao "was very agile. He could play inside, he could play outside, and if he didn't have those injuries he could've been up there in the top five centers to ever play the game."[123]
Yao was nominated by a member of the Chinese media for the Naismith Basketball Hall of Fame as a contributor to the game. He would have been eligible for induction as early as 2012, but Yao felt it was too soon and requested that the Hall of Fame delay consideration of the nomination. The Hall granted Yao's request, and said it was Yao's decision when the process would be restarted.[124]
On September 9, 2016, Yao was inducted into the Hall of Fame along with 4-time NBA champion Shaquille O'Neal and Allen Iverson.[125] Continuing with the honors, on February 3, 2017, Yao's Number 11 jersey was retired by the Houston Rockets.[126]

"""

# Demo configuration - modify these as needed
question = chinese_question
context = chinese_context
max_seq_len = 4096 
compression_rate = 0.9
context_type = "chinese"  # changed from "chinese" to match content

# Auto-detect device
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def demo_raw_attention():
    """Demo raw attention filtering without external detector"""
    print("=== Raw Attention Filtering Demo ===")
    
    compressor = AttentionCompressor(
        attention_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        use_raw_attention=True,  # No detector needed
        use_last_layer_only=False,  # Use all layers
        use_all_queries=False,  # Use last token only
        max_seq_len=max_seq_len,
        device=device
    )
    
    result = compressor.compress(
        context=context,
        question=question,
        target_token=-1,  # Use compression_rate instead of fixed token count
        compression_rate=compression_rate,
        context_type=context_type
    )
    
    print(f"Original length: {result['original_length']} tokens")
    print(f"Compressed length: {result['compressed_length']} tokens")
    print(f"Compression ratio: {result['compression_ratio']:.2%}")
    print(f"Processing time: {result['processing_time']:.3f}s")
    print("\nOriginal text (first 200 chars):")
    print(context[:200] + "...")
    print("\nCompressed text:")
    print(result['compressed_text'])
    print()


def demo_detector_based():
    """Demo detector-based filtering (requires trained detector)"""
    print("=== Detector-Based Filtering Demo ===")
    
    # Build detector path
    detector_path = os.path.join(DETECTOR_BASE_PATH, DETECTOR_CONFIGS["standard"])
    
    try:
        print(f"Using detector: {detector_path}")
        if not os.path.exists(detector_path):
            print(f"⚠️  Detector model not found at: {detector_path}")
            print("   Please update DETECTOR_BASE_PATH and DETECTOR_CONFIGS in the configuration section")
            return
        compressor = AttentionCompressor(
            attention_model_path="Qwen/Qwen2.5-0.5B-Instruct",
            detector_path=detector_path,
            use_raw_attention=False,  # Use detector
            use_last_layer_only=False,
            use_all_queries=False,
            max_seq_len=max_seq_len,
            device=device
        )
        
        result = compressor.compress(
            context=context,
            question=question,
            target_token=-1,  # Use compression_rate instead of fixed token count
            compression_rate=compression_rate,  # Use consistent compression rate
            context_type=context_type
        )
        
        print(f"Original length: {result['original_length']} tokens")
        print(f"Compressed length: {result['compressed_length']} tokens")
        print(f"Compression ratio: {result['compression_ratio']:.2%}")
        print(f"Processing time: {result['processing_time']:.3f}s")
        print("\nOriginal text (first 200 chars):")
        print(context[:200] + "...")
        print("\nCompressed text:")
        print(result['compressed_text'])
        print()
        
    except Exception as e:
        print(f"❌ Detector demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def demo_selected_feature_idx():
    """Demo selected feature idx mode with specialized detector"""
    print("=== Selected Feature Idx Demo ===")
    
    # Build selected feature idx detector path
    detector_path = os.path.join(DETECTOR_BASE_PATH, DETECTOR_CONFIGS["selected_idx"])
    
    try:
        print(f"Using selected feature idx detector: {detector_path}")
        if not os.path.exists(detector_path):
            print(f"⚠️  Selected idx detector model not found at: {detector_path}")
            print("   Please update DETECTOR_BASE_PATH and DETECTOR_CONFIGS in the configuration section")
            return
        compressor = AttentionCompressor(
            attention_model_path="Qwen/Qwen2.5-0.5B-Instruct",
            detector_path=detector_path,
            use_raw_attention=False,  # Use detector
            use_last_layer_only=False,  # Use all layers
            use_all_queries=False,  # Use last token
            max_seq_len=max_seq_len,
            device=device,
            do_selected_feature_idx=True  # Enable selected feature idx mode
        )
        
        result = compressor.compress(
            context=context,
            question=question,
            target_token=-1,  # Use compression_rate instead of fixed token count
            compression_rate= compression_rate,  # Use consistent compression rate
            context_type=context_type
        )
        
        print(f"Original length: {result['original_length']} tokens")
        print(f"Compressed length: {result['compressed_length']} tokens")
        print(f"Compression ratio: {result['compression_ratio']:.2%}")
        print(f"Processing time: {result['processing_time']:.3f}s")
        print("\nOriginal text (first 200 chars):")
        print(context[:200] + "...")
        print("\nCompressed text:")
        print(result['compressed_text'])
        print()
        
    except Exception as e:
        print(f"❌ Selected feature idx demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def main():
    """Run all demos"""
    print("Attention-Based Text Compression Demos")
    print("=" * 50)
    print(f"🖥️  Using device: {device}")
    print(f"📄  Context type: {context_type}")
    print(f"🗜️  Compression rate: {compression_rate:.1%}")
    print("=" * 50)
    
    try:
        demo_raw_attention()
        demo_detector_based()
        # demo_selected_feature_idx()
        print("🎉 All demos completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 

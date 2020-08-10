CREATE TABLE RConversations (
    Convo_num int NOT NULL AUTO_INCREMENT,
    Client_ID varchar(8),
    WS_Version varchar(16),
    First_name varchar(255),
    Last_name varchar(255),
    Patient_choice int,
    Input_method varchar(8),
    Mic varchar(8),
    Exp_group varchar(16),
    Raw_score TEXT,
    Uuid varchar(40),
    PRIMARY KEY (Convo_num)
);

CREATE TABLE RQueries (
    Convo_num int NOT NULL,
    Query_num int NOT NULL,
    Input_text varchar(510),
    CS_interp varchar(510),
    RNN_interp varchar(510),
    CS_init_reply varchar(510),
    CS_retry_reply varchar(510),
    Choice varchar(8),
    Audio_path varchar(255),
    CONSTRAINT RPK_Query PRIMARY KEY (Convo_num, Query_num),
    CONSTRAINT RFK_Query_Convo FOREIGN KEY (Convo_num) REFERENCES RConversations(Convo_num)
);
					

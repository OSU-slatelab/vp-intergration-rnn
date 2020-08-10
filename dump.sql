SELECT RConversations.Convo_num,
       RQueries.Query_num,
       RConversations.Client_ID,
       RConversations.WS_Version,
       RConversations.First_name,
       RConversations.Last_name,
       RConversations.Patient_choice,
       RQueries.Input_text,
       RQueries.CS_interp,
       RQueries.RNN_interp,
       RQueries.CS_init_reply,
       RQueries.CS_retry_reply,
       RQueries.Choice,
       RQueries.Audio_path
FROM RConversations
     JOIN RQueries ON RConversations.Convo_num = RQueries.Convo_num
WHERE RConversations.Convo_num > 1055;
